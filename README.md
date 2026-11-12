# 💳 Credit Risk Scoring & Underwriting Service

Production-ready ML-сервис для оценки кредитного риска заемщиков (Credit Scoring) и прогнозирования вероятности дефолта с модулем интерпретации решений (SHAP Reason Codes).

---

## 📌 Бизнес-контекст задачи

Банку необходимо принимать решения по выдаче потребительских кредитов:
* **Ошибка I рода (False Positive)**: отказ надежному заемщику. Банк теряет потенциальную прибыль с процентов (упущенная выгода).
* **Ошибка II рода (False Negative)**: выдача кредита клиенту, который уйдет в дефолт. Банк несет прямые убытки от невозврата основного долга (в 5–10 раз дороже ошибки I рода).

Стандартный порог классификации `0.5` и метрика `Accuracy` на несбалансированных данных (всего **6.68%** дефолтов в выборке) неприменимы. В проекте реализован полный цикл: от устранения утечек данных до оптимизации порога под бизнес-матрицу потерь и упаковки в REST API.

---

## 📊 Результаты и сравнение моделей

Оценка моделей проводилась на отложенной тестовой выборке (30 000 анкет, 20% стратифицированный сплит):

| Модель | ROC-AUC | Gini ($2 \cdot \text{AUC} - 1$) | PR-AUC | Brier Score | Recall (дефолты) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression + Balanced)** | 0.8594 | 0.7188 | 0.3367 | 0.1464 | 73.6% |
| **LightGBM + Domain Feature Engineering** | **0.8698** | **0.7396** | **0.4070** | **0.1382** | **77.3%** |

### Экономический эффект подбора порога (Cost Matrix)
При соотношении стоимости ошибок $Cost_{FN} : Cost_{FP} = 8 : 1$:
* **Дефолтный порог (0.50)**: общие финансовые потери = **9 069** у.е.
* **Оптимальный порог (0.63)**: общие финансовые потери = **8 463** у.е.
* **Результат**: снижение потерь банка на **~6.7%** и одобрение на **2 294** больше надежных заемщиков без роста риска дефолта.

---

## 🛠 Ключевые инженерные решения

1. **Защита от утечек данных (Zero Leakage)**:
   - Все расчеты медиан, скейлинг и генерация признаков инкапсулированы в `CreditFeatureEngineer` и `scikit-learn.Pipeline`.
   - Вызов `.fit()` выполняется строго на обучающей выборке (`train.parquet`).
2. **Доменный Feature Engineering**:
   - `Missing_MonthlyIncome` / `Missing_NumberOfDependents` — индикаторы отсутствия дохода в анкете.
   - `TotalLateCount` и `HasLate90` — агрегация фактов просрочки с фильтрацией служебных кодов БКИ (96/98).
   - `IncomePerPerson` — подушевой доход на одного члена семьи.
   - `UtilizationOverLimit` — маркер превышения лимита кредитных карт.
3. **Объяснимость решений (Explainability)**:
   - По требованиям регуляторов банк обязан формулировать причину отказа (Adverse Action Notice). В сервис встроен `TreeExplainer` на базе SHAP, возвращающий топ-3 фактора риска в понятном виде на русском языке.
4. **Надежность сервиса (Serving & Quality)**:
   - Строгая валидация входящих типов и диапазонов через `Pydantic v2`.
   - Полное покрытие пайплайна и API модульными и интеграционными тестами (`pytest`).

---

## 📁 Структура репозитория

```text
├── configs/
│   └── config.yaml              # Гиперпараметры, пути и бизнес-коэффициенты
├── data/                        # Сырые и обработанные данные (.gitignored)
│   ├── raw/                     # Исходный датасет cs-training.csv
│   └── processed/               # train.parquet и test.parquet
├── models/                      # Сохраненный пайплайн и артефакты метрик
│   ├── best_model.joblib        # Обученный пайплайн (препроцессинг + LightGBM)
│   └── metrics.json             # Итоговые метрики на тестовой выборке
├── notebooks/
│   └── 01_eda.ipynb             # Разведочный анализ данных и обоснование гипотез
├── src/
│   ├── data/
│   │   └── make_dataset.py      # Загрузка и стратифицированное разбиение данных
│   ├── features/
│   │   └── build_features.py    # Кастомные трансформеры Scikit-Learn
│   ├── models/
│   │   ├── metrics.py           # Расчет Gini, кредитного скора и поиск порога
│   │   ├── train.py             # Скрипт обучения и валидации моделей
│   │   └── explainability.py    # Модуль расчета SHAP reason codes
│   └── api/
│       ├── schemas.py           # Pydantic схемы валидации запросов и ответов
│       └── app.py               # FastAPI веб-сервис
├── tests/
│   ├── test_features.py         # Тесты трансформаций признаков и отсутствия NaN
│   └── test_api.py              # Тесты эндпоинтов, валидации и сценариев скоринга
├── Dockerfile                   # Контейнеризация сервиса
├── requirements.txt             # Зависимости проекта
└── README.md
```

---

## 🚀 Быстрый запуск

### 1. Локальная установка и запуск

```bash
# Клонирование репозитория
git clone https://github.com/ClipFailer/credit-risk-scoring.git
cd credit-risk-scoring

# Установка зависимостей
pip install -r requirements.txt

# Скачивание данных и подготовка train/test сплита
python -m src.data.make_dataset

# Запуск обучения моделей и расчет метрик
python -m src.models.train

# Запуск тестов
pytest -v

# Старт API-сервера
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Интерактивная документация Swagger UI доступна по адресу: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2. Запуск через Docker

```bash
# Сборка образа
docker build -t credit-scoring-api:latest .

# Запуск контейнера
docker run -p 8000:8000 credit-scoring-api:latest
```

---

## 📡 Пример вызова API (`/predict`)

### Запрос (Анкета надежного заемщика):

```json
POST /predict
Content-Type: application/json

{
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
```

### Ответ сервиса:

```json
{
  "probability_of_default": 0.0312,
  "credit_score": 833,
  "decision": "APPROVED",
  "risk_category": "LOW",
  "threshold_used": 0.6305,
  "reason_codes": [
    {
      "feature": "RevolvingUtilizationOfUnsecuredLines",
      "description": "Высокая доля использования кредитного лимита",
      "shap_impact": -0.8421,
      "original_value": 0.08
    },
    {
      "feature": "DebtRatio",
      "description": "Высокий уровень долговой нагрузки (DTI)",
      "shap_impact": -0.4125,
      "original_value": 0.18
    }
  ]
}
```
