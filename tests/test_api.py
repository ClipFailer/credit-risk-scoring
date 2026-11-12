"""Интеграционные тесты для FastAPI сервиса скоринга."""

from fastapi.testclient import TestClient
import pytest

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """Тестовый клиент FastAPI с предварительной загрузкой состояния (lifespan)."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Проверка доступности сервиса."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    """Проверка эндпоинта информации о модели."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "business_parameters" in data


def test_predict_prime_borrower(client):
    """Надежный заемщик должен получить одобрение (APPROVED) и высокий кредитный скор."""
    prime_payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.08,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.18,
        "MonthlyIncome": 9500.0,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0
    }

    response = client.post("/predict", json=prime_payload)
    assert response.status_code == 200
    result = response.json()

    assert result["decision"] == "APPROVED"
    assert result["credit_score"] >= 650
    assert result["probability_of_default"] < 0.3
    assert len(result["reason_codes"]) > 0


def test_predict_risky_borrower(client):
    """Рискованный заемщик (просрочки, превышение лимитов) должен получить отказ (REJECTED)."""
    risky_payload = {
        "RevolvingUtilizationOfUnsecuredLines": 1.45,
        "age": 24,
        "NumberOfTime30-59DaysPastDueNotWorse": 3,
        "DebtRatio": 0.95,
        "MonthlyIncome": 1200.0,
        "NumberOfOpenCreditLinesAndLoans": 2,
        "NumberOfTimes90DaysLate": 2,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 0.0
    }

    response = client.post("/predict", json=risky_payload)
    assert response.status_code == 200
    result = response.json()

    assert result["decision"] == "REJECTED"
    assert result["credit_score"] < 650
    assert result["probability_of_default"] > 0.4
    assert len(result["reason_codes"]) > 0


def test_predict_validation_error(client):
    """Некорректный возраст (< 18 лет) должен возвращать ошибку валидации 422."""
    invalid_payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.2,
        "age": 15,  # Минимальный допустимый возраст - 18
        "DebtRatio": 0.3,
        "NumberOfOpenCreditLinesAndLoans": 1
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
