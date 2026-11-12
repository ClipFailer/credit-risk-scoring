"""Загрузка сырых данных, первичная очистка и стратифицированное разбиение."""

import logging
from pathlib import Path
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("make_dataset")

DATA_URL = "https://raw.githubusercontent.com/liuyizhi123/give-me-some-credit-master/master/data/cs-training.csv"


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Загрузка конфигурации проекта."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_raw_data(dest_path: str, url: str = DATA_URL) -> Path:
    """Скачивание датасета, если он еще не сохранен локально."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 100_000:
        logger.info(f"Сырые данные уже существуют: {dest}")
        return dest

    logger.info(f"Скачиваем датасет по адресу: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f_out:
        f_out.write(resp.read())

    logger.info(f"Файл успешно сохранен: {dest} ({dest.stat().st_size / (1024*1024):.2f} MB)")
    return dest


def prepare_and_split_data(config_path: str = "configs/config.yaml") -> None:
    """Загрузка данных и разбиение на train/test со стратификацией по таргету."""
    config = load_config(config_path)

    raw_path = config["data"]["raw_path"]
    train_path = Path(config["data"]["processed_train_path"])
    test_path = Path(config["data"]["processed_test_path"])
    target_col = config["data"]["target_column"]
    test_size = config["data"]["test_size"]
    seed = config["project"]["random_state"]

    download_raw_data(raw_path)

    df = pd.read_csv(raw_path)

    # Убираем лишний безымянный столбец с индексами из Kaggle
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    logger.info(f"Размер исходного датасета: {df.shape[0]} строк, {df.shape[1]} признаков")

    # Проверяем долю дефолтов для контроля баланса
    default_share = df[target_col].mean() * 100
    logger.info(f"Доля дефолтов в выборке: {default_share:.2f}%")

    # Стратифицированный сплит: сохраняем пропорцию редкого класса в train и test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=seed
    )

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем в parquet для быстрого чтения и сжатия
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logger.info(f"Обучающая выборка: {train_path} ({len(train_df)} строк)")
    logger.info(f"Тестовая выборка: {test_path} ({len(test_df)} строк)")


if __name__ == "__main__":
    prepare_and_split_data()
