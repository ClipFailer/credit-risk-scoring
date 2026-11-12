"""Pydantic-схемы для валидации входных данных анкеты и ответа сервиса скоринга."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LoanApplicationRequest(BaseModel):
    """Данные кредитной анкеты заемщика."""

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ...,
        description="Отношение баланса кредитных карт к общему лимиту",
        ge=0.0,
        examples=[0.35]
    )
    age: int = Field(
        ...,
        description="Возраст заемщика (полных лет)",
        ge=18,
        le=120,
        examples=[38]
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        default=0,
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        description="Количество просрочек от 30 до 59 дней за последние 2 года",
        ge=0,
        examples=[0]
    )
    DebtRatio: float = Field(
        ...,
        description="Коэффициент долговой нагрузки (DTI)",
        ge=0.0,
        examples=[0.28]
    )
    MonthlyIncome: Optional[float] = Field(
        default=None,
        description="Ежемесячный доход",
        ge=0.0,
        examples=[5500.0]
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ...,
        description="Количество открытых кредитов и кредитных карт",
        ge=0,
        examples=[5]
    )
    NumberOfTimes90DaysLate: int = Field(
        default=0,
        description="Количество серьезных просрочек на 90+ дней",
        ge=0,
        examples=[0]
    )
    NumberRealEstateLoansOrLines: int = Field(
        default=0,
        description="Количество ипотечных кредитов",
        ge=0,
        examples=[1]
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        default=0,
        alias="NumberOfTime60-89DaysPastDueNotWorse",
        description="Количество просрочек от 60 до 89 дней за последние 2 года",
        ge=0,
        examples=[0]
    )
    NumberOfDependents: Optional[float] = Field(
        default=0.0,
        description="Количество иждивенцев в семье",
        ge=0.0,
        examples=[1.0]
    )

    model_config = {
        "populate_by_name": True
    }


class ReasonCode(BaseModel):
    """Фактор, повлиявший на оценку кредитного риска."""
    feature: str
    description: str
    shap_impact: float
    original_value: Optional[float] = None


class CreditScoringResponse(BaseModel):
    """Результат скоринговой оценки кредитной заявки."""

    probability_of_default: float = Field(..., description="Вероятность дефолта (PD)")
    credit_score: int = Field(..., description="Скоринговый балл от 300 до 850")
    decision: str = Field(..., description="Решение: APPROVED или REJECTED")
    risk_category: str = Field(..., description="Категория риска: LOW, MEDIUM, HIGH")
    threshold_used: float = Field(..., description="Использованный порог отсечки")
    reason_codes: List[ReasonCode] = Field(..., description="Топ факторов риска по SHAP")


class ModelInfoResponse(BaseModel):
    """Информация об активной модели и параметрах сервиса."""
    model_name: str
    algorithm: str
    metrics: Dict[str, Any]
    business_parameters: Dict[str, Any]
