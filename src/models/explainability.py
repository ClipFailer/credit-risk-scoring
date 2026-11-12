"""Модуль объяснения предсказаний (SHAP Reason Codes)."""

import logging
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger("explainability")

# Человекочитаемые описания факторов риска для формирования отчета/ответа API
FEATURE_RU_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Высокая доля использования кредитного лимита",
    "NumberOfTimes90DaysLate": "Наличие серьезных просрочек 90+ дней",
    "TotalLateCount": "Общее количество фактов просрочки в кредитной истории",
    "DebtRatio": "Высокий уровень долговой нагрузки (DTI)",
    "age": "Возраст заемщика",
    "NumberOfTime30-59DaysPastDueNotWorse": "Краткосрочные просрочки (30-59 дней)",
    "NumberOfTime60-89DaysPastDueNotWorse": "Среднесрочные просрочки (60-89 дней)",
    "MonthlyIncome": "Уровень ежемесячного дохода",
    "NumberOfOpenCreditLinesAndLoans": "Количество открытых кредитов и кредитных карт",
    "NumberRealEstateLoansOrLines": "Количество ипотечных кредитов",
    "IncomePerPerson": "Доход на одного члена семьи",
    "UtilizationOverLimit": "Превышение установленного кредитного лимита",
    "Missing_MonthlyIncome": "Отсутствие подтвержденного дохода в анкете"
}


class CreditRiskExplainer:
    """Интерпретация предсказаний модели скоринга с помощью значений SHAP."""

    def __init__(self, model_pipeline_path: str = "models/best_model.joblib"):
        self.pipeline = joblib.load(model_pipeline_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.explainer = shap.TreeExplainer(self.classifier)

    def explain_application(self, applicant_df: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
        """Определяет топ-K факторов, оказавших наибольшее влияние на риск дефолта."""
        transformed = self.preprocessor.transform(applicant_df)
        feature_names = transformed.columns.tolist()

        shap_vals = self.explainer.shap_values(transformed)

        # Приводим вывод к одномерному массиву для одной анкеты
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            row_vals = shap_vals[1][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
            row_vals = shap_vals[0]
        else:
            row_vals = np.array(shap_vals).flatten()[:len(feature_names)]

        # Сортируем признаки по вкладу в повышение вероятности дефолта
        ranked_indices = np.argsort(row_vals)[::-1]
        reasons = []

        for idx in ranked_indices[:top_k]:
            fname = feature_names[idx]
            val = float(row_vals[idx])
            desc = FEATURE_RU_NAMES.get(fname, fname)
            original_val = applicant_df[fname].values[0] if fname in applicant_df.columns else None

            reasons.append({
                "feature": fname,
                "description": desc,
                "shap_impact": round(val, 4),
                "original_value": round(float(original_val), 2) if original_val is not None and not pd.isna(original_val) else None
            })

        return reasons
