"""Тестирование генерации признаков и устойчивости пайплайна."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import CreditFeatureEngineer


@pytest.fixture
def sample_data():
    """Синтетический мини-набор данных для проверки трансформаций."""
    return pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [0.35, 1.5, 0.1],
        "age": [35, 0, 50],
        "NumberOfTime30-59DaysPastDueNotWorse": [0, 98, 1],
        "DebtRatio": [0.25, 0.8, 0.15],
        "MonthlyIncome": [5000.0, np.nan, 8000.0],
        "NumberOfOpenCreditLinesAndLoans": [4, 2, 8],
        "NumberOfTimes90DaysLate": [0, 98, 0],
        "NumberRealEstateLoansOrLines": [1, 0, 2],
        "NumberOfTime60-89DaysPastDueNotWorse": [0, 96, 0],
        "NumberOfDependents": [1.0, np.nan, 2.0]
    })


def test_feature_engineer_output_columns(sample_data):
    """Проверка создания всех обязательных доменных признаков."""
    transformer = CreditFeatureEngineer()
    transformed = transformer.fit_transform(sample_data)

    expected_cols = [
        "Missing_MonthlyIncome",
        "Missing_NumberOfDependents",
        "TotalLateCount",
        "HasAnyLate",
        "HasLate90",
        "UtilizationOverLimit",
        "IncomePerPerson"
    ]
    for col in expected_cols:
        assert col in transformed.columns, f"Признак {col} не найден в выводе трансформера"


def test_no_nan_values_after_transform(sample_data):
    """Трансформер не должен оставлять NaN в итоговой таблице."""
    transformer = CreditFeatureEngineer()
    transformed = transformer.fit_transform(sample_data)
    assert transformed.isna().sum().sum() == 0, "После transform остались пропуски (NaN)"


def test_sentinel_codes_clipping(sample_data):
    """Служебные коды 96/98 должны корректно ограничиваться."""
    transformer = CreditFeatureEngineer()
    transformed = transformer.fit_transform(sample_data)

    assert (transformed["NumberOfTimes90DaysLate"] <= 20).all()
    assert (transformed["NumberOfTime30-59DaysPastDueNotWorse"] <= 20).all()
    assert (transformed["NumberOfTime60-89DaysPastDueNotWorse"] <= 20).all()


def test_age_anomaly_correction(sample_data):
    """Возраст <= 0 должен заменяться на медианный возраст выборки."""
    transformer = CreditFeatureEngineer()
    transformed = transformer.fit_transform(sample_data)

    assert (transformed["age"] > 0).all()
