"""Утилиты работы с внешней базой Face ID сотрудников.

Зависимости::
    pip install numpy pandas

Формат ожидаемой базы может быть расширен при интеграции с реальной системой.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def load_face_db(base_dir: str | Path = "face_db") -> Dict[str, dict]:
    """Загружает локальную базу сотрудников с эмбеддингами лиц.

    Ожидается наличие файла ``embeddings.npz`` с массивом эмбеддингов и
    файла ``metadata.csv`` с колонками ``employee_id`` и ``name``. Формат может
    быть адаптирован под конкретный источник. Если файлы отсутствуют, возвращает
    пустой словарь.
    """

    base_path = Path(base_dir)
    embeddings_path = base_path / "embeddings.npz"
    metadata_path = base_path / "metadata.csv"

    if not embeddings_path.exists() or not metadata_path.exists():
        return {}

    metadata_df = pd.read_csv(metadata_path)
    embeddings_data = np.load(embeddings_path)

    face_db: Dict[str, dict] = {}
    for _, row in metadata_df.iterrows():
        employee_id = str(row["employee_id"])
        embedding_key = f"embedding_{employee_id}"
        embedding = embeddings_data.get(embedding_key)
        if embedding is None:
            continue
        face_db[employee_id] = {
            "name": str(row.get("name", "")),
            "embedding": np.asarray(embedding, dtype=np.float32),
        }

    return face_db


def save_face_db(
    face_db: Dict[str, dict], base_dir: str | Path = "face_db"
) -> None:
    """Сохраняет базу сотрудников на диск.

    Эта функция демонстрирует ожидаемый формат. В реальном проекте возможно
    чтение/запись из внешних систем. Эмбеддинги сохраняются в ``embeddings.npz``,
    метаданные — в ``metadata.csv``.
    """

    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, np.ndarray] = {}
    rows = []
    for employee_id, info in face_db.items():
        embedding = np.asarray(info.get("embedding", []), dtype=np.float32)
        embeddings[f"embedding_{employee_id}"] = embedding
        rows.append({"employee_id": employee_id, "name": info.get("name", "")})

    np.savez(base_path / "embeddings.npz", **embeddings)
    pd.DataFrame(rows).to_csv(base_path / "metadata.csv", index=False)
