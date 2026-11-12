"""Обучение и сравнение моделей: линейный бейзлайн против LightGBM."""

import json
import logging
from pathlib import Path
import joblib
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import yaml

from src.features.build_features import CreditFeatureEngineer, create_preprocessing_pipeline
from src.models.metrics import evaluate_predictions, find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Загрузка конфигурации проекта."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_models(config_path: str = "configs/config.yaml") -> dict:
    """Обучение моделей, расчет метрик на тесте и выбор лучшего пайплайна."""
    config = load_config(config_path)

    train_path = config["data"]["processed_train_path"]
    test_path = config["data"]["processed_test_path"]
    target_col = config["data"]["target_column"]
    seed = config["project"]["random_state"]
    cost_fp = config["business"]["cost_false_positive"]
    cost_fn = config["business"]["cost_false_negative"]
    output_dir = Path(config["model"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Загрузка train и test выборок...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    results = {}

    # 1. Бейзлайн: логистическая регрессия с балансировкой весов классов
    logger.info("Обучение бейзлайна (Logistic Regression)...")
    baseline_pipe = Pipeline([
        ("preprocessor", create_preprocessing_pipeline()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed))
    ])

    baseline_pipe.fit(X_train, y_train)
    y_prob_base = baseline_pipe.predict_proba(X_test)[:, 1]
    results["Baseline_LogisticRegression"] = evaluate_predictions(y_test, y_prob_base)
    logger.info(
        f"Бейзлайн | ROC-AUC: {results['Baseline_LogisticRegression']['roc_auc']:.4f} | "
        f"Gini: {results['Baseline_LogisticRegression']['gini']:.4f}"
    )

    # 2. Основная модель: градиентный бустинг LightGBM
    logger.info("Обучение основной модели (LightGBM)...")
    lgb_preprocessor = Pipeline([
        ("feature_engineer", CreditFeatureEngineer())
    ])

    lgb_pipe = Pipeline([
        ("preprocessor", lgb_preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            class_weight="balanced",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1
        ))
    ])

    lgb_pipe.fit(X_train, y_train)
    y_prob_lgb = lgb_pipe.predict_proba(X_test)[:, 1]
    results["LightGBM"] = evaluate_predictions(y_test, y_prob_lgb)
    logger.info(
        f"LightGBM  | ROC-AUC: {results['LightGBM']['roc_auc']:.4f} | "
        f"Gini: {results['LightGBM']['gini']:.4f}"
    )

    # 3. Подбор оптимального бизнес-порога с учетом матрицы потерь
    logger.info("Поиск оптимального порога по матрице затрат...")
    threshold_opt = find_optimal_threshold(
        y_test,
        y_prob_lgb,
        cost_fp=cost_fp,
        cost_fn=cost_fn
    )
    results["Threshold_Optimization"] = threshold_opt
    logger.info(
        f"Оптимальный порог: {threshold_opt['threshold']} (общие потери: {threshold_opt['total_cost']})"
    )

    # 4. Сохраняем модель и метрики
    model_path = output_dir / config["model"]["model_filename"]
    joblib.dump(lgb_pipe, model_path)
    logger.info(f"Лучшая модель сохранена: {model_path}")

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Метрики сохранены: {metrics_path}")

    return results


if __name__ == "__main__":
    train_models()
