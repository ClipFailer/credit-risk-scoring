"""REST API сервис для онлайн-скоринга кредитных заявок."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, status

from src.api.schemas import (
    CreditScoringResponse,
    LoanApplicationRequest,
    ModelInfoResponse,
    ReasonCode
)
from src.models.explainability import CreditRiskExplainer
from src.models.metrics import calculate_credit_score

# Состояние приложения (кэширование пайплайна и эксплейнера в памяти)
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация модели и вспомогательных объектов при запуске сервиса."""
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        raise RuntimeError("Конфигурационный файл configs/config.yaml не найден.")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = Path(config["model"]["output_dir"]) / config["model"]["model_filename"]
    if not model_path.exists():
        raise RuntimeError(f"Файл модели не найден по пути {model_path}. Сначала запустите обучение.")

    metrics_path = Path(config["model"]["output_dir"]) / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    # Используем оптимальный порог, рассчитанный по матрице потерь
    optimal_threshold = metrics.get("Threshold_Optimization", {}).get("threshold", config["business"]["default_threshold"])

    app_state["config"] = config
    app_state["model"] = joblib.load(model_path)
    app_state["explainer"] = CreditRiskExplainer(str(model_path))
    app_state["metrics"] = metrics
    app_state["threshold"] = optimal_threshold

    yield
    app_state.clear()


app = FastAPI(
    title="Сервис Кредитного Скоринга",
    description="ML-сервис скоринга заемщиков в реальном времени с поддержкой интерпретации решений (SHAP).",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Служебные"])
def health_check():
    """Проверка доступности сервиса и статуса загрузки модели."""
    return {
        "status": "healthy",
        "model_loaded": "model" in app_state
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Модель"])
def get_model_info():
    """Получение метрик качества модели и рабочих параметров скоринга."""
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Модель не загружена.")

    return {
        "model_name": "CreditRisk-LightGBM",
        "algorithm": "LightGBM + Feature Engineering",
        "metrics": app_state["metrics"],
        "business_parameters": {
            "optimal_threshold": app_state["threshold"],
            "score_range": [app_state["config"]["business"]["score_min"], app_state["config"]["business"]["score_max"]],
            "cost_fp": app_state["config"]["business"]["cost_false_positive"],
            "cost_fn": app_state["config"]["business"]["cost_false_negative"]
        }
    }


@app.post("/predict", response_model=CreditScoringResponse, tags=["Скоринг"])
def score_applicant(application: LoanApplicationRequest):
    """Скоринг кредитной заявки: расчет вероятности дефолта, скорингового балла и факторов отказа."""
    if "model" not in app_state:
        raise HTTPException(status_code=503, detail="Модель не загружена.")

    # Преобразуем входящий payload в датафрейм с корректными именами колонок
    payload_dict = application.model_dump(by_alias=True)
    df = pd.DataFrame([payload_dict])

    pipeline = app_state["model"]
    explainer = app_state["explainer"]
    threshold = app_state["threshold"]

    # Расчет вероятности дефолта (PD)
    prob = float(pipeline.predict_proba(df)[0, 1])
    score = calculate_credit_score(prob)

    # Принятие решения по оптимальному порогу
    decision = "REJECTED" if prob >= threshold else "APPROVED"

    # Категория риска
    if prob < 0.2:
        risk_tier = "LOW"
    elif prob < 0.5:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "HIGH"

    # Формируем топ-3 фактора риска через SHAP
    raw_reasons = explainer.explain_application(df, top_k=3)
    reasons = [ReasonCode(**r) for r in raw_reasons]

    return CreditScoringResponse(
        probability_of_default=round(prob, 4),
        credit_score=score,
        decision=decision,
        risk_category=risk_tier,
        threshold_used=round(threshold, 4),
        reason_codes=reasons
    )
