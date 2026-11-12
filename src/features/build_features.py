"""Пайплайн предобработки и генерации банковских признаков."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Генерация доменных признаков для кредитного скоринга.

    Статистики (медианы) считаются строго на этапе fit, чтобы исключить утечки данных.
    """

    def __init__(self):
        self.median_income_ = None
        self.median_dependents_ = None
        self.median_age_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Считаем медианы для безопасного заполнения пропусков в transform
        self.median_income_ = float(X["MonthlyIncome"].median()) if "MonthlyIncome" in X.columns else 0.0
        self.median_dependents_ = float(X["NumberOfDependents"].median()) if "NumberOfDependents" in X.columns else 0.0
        self.median_age_ = float(X["age"].median()) if "age" in X.columns else 45.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if not isinstance(X_out, pd.DataFrame):
            X_out = pd.DataFrame(X_out)

        # Флаги отсутствия данных (в скоринге факт пропуска дохода сам по себе несет сигнал)
        if "MonthlyIncome" in X_out.columns:
            X_out["Missing_MonthlyIncome"] = X_out["MonthlyIncome"].isna().astype(float)
        if "NumberOfDependents" in X_out.columns:
            X_out["Missing_NumberOfDependents"] = X_out["NumberOfDependents"].isna().astype(float)

        # Корректируем аномальные значения возраста (например, 0 лет)
        if "age" in X_out.columns:
            X_out.loc[X_out["age"] <= 0, "age"] = self.median_age_

        # Обработка истории просрочек
        # Значения 96 и 98 в датасете — это служебные коды бюро кредитных историй (ошибки/нет данных).
        # Ограничиваем их верхнюю границу, чтобы они не ломали веса линейных моделей
        past_due_cols = [
            "NumberOfTime30-59DaysPastDueNotWorse",
            "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"
        ]
        present_cols = [c for c in past_due_cols if c in X_out.columns]
        if present_cols:
            for c in present_cols:
                X_out[c] = X_out[c].clip(upper=20)

            # Суммарное количество просрочек любого типа
            X_out["TotalLateCount"] = X_out[present_cols].sum(axis=1)
            # Бинарный флаг: были ли просрочки вообще
            X_out["HasAnyLate"] = (X_out["TotalLateCount"] > 0).astype(float)

        # Отдельный флаг критической просрочки (90+ дней)
        if "NumberOfTimes90DaysLate" in X_out.columns:
            X_out["HasLate90"] = (X_out["NumberOfTimes90DaysLate"] > 0).astype(float)

        # Утилизация кредитных линий
        if "RevolvingUtilizationOfUnsecuredLines" in X_out.columns:
            # Превышение кредитного лимита (> 100%)
            X_out["UtilizationOverLimit"] = (X_out["RevolvingUtilizationOfUnsecuredLines"] > 1.0).astype(float)
            # Ограничиваем выбросы
            X_out["RevolvingUtilizationOfUnsecuredLines"] = X_out["RevolvingUtilizationOfUnsecuredLines"].clip(upper=10.0)

        # Расчет подушевого дохода на члена семьи
        income = X_out["MonthlyIncome"].fillna(self.median_income_) if "MonthlyIncome" in X_out.columns else 0.0
        deps = X_out["NumberOfDependents"].fillna(self.median_dependents_) if "NumberOfDependents" in X_out.columns else 0.0
        X_out["IncomePerPerson"] = income / (deps + 1.0)

        # Заполняем пропуски сохраненными медианами
        if "MonthlyIncome" in X_out.columns:
            X_out["MonthlyIncome"] = X_out["MonthlyIncome"].fillna(self.median_income_)
        if "NumberOfDependents" in X_out.columns:
            X_out["NumberOfDependents"] = X_out["NumberOfDependents"].fillna(self.median_dependents_)

        # Страховка от возможных оставшихся NaN
        X_out = X_out.fillna(0.0)
        return X_out


def create_preprocessing_pipeline() -> Pipeline:
    """Сборка пайплайна предобработки для линейного бейзлайна."""
    return Pipeline([
        ("feature_engineer", CreditFeatureEngineer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])
