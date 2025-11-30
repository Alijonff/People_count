"""
Утилиты для работы с базой лиц (Known Employees).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import os

def load_face_db(base_dir: str | Path = "face_db") -> Dict[str, dict]:
    """
    Загружает базу сотрудников.
    Возвращает словарь: { "employee_id": { "name": str, "embedding": np.ndarray } }
    """
    base_path = Path(base_dir)
    embeddings_path = base_path / "embeddings.npz"
    metadata_path = base_path / "metadata.csv"

    if not embeddings_path.exists() or not metadata_path.exists():
        return {}

    try:
        metadata_df = pd.read_csv(metadata_path)
        embeddings_data = np.load(embeddings_path)
    except Exception as e:
        print(f"Ошибка загрузки базы лиц: {e}")
        return {}

    face_db: Dict[str, dict] = {}
    for _, row in metadata_df.iterrows():
        employee_id = str(row["employee_id"])
        embedding_key = f"embedding_{employee_id}"
        if embedding_key in embeddings_data:
            face_db[employee_id] = {
                "name": str(row.get("name", "")),
                "embedding": np.asarray(embeddings_data[embedding_key], dtype=np.float32),
            }
    return face_db

def save_face_db(face_db: Dict[str, dict], base_dir: str | Path = "face_db") -> None:
    """
    Сохраняет базу сотрудников.
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    embeddings_dict = {}
    metadata_rows = []

    for employee_id, data in face_db.items():
        embeddings_dict[f"embedding_{employee_id}"] = data["embedding"]
        metadata_rows.append({
            "employee_id": employee_id,
            "name": data["name"]
        })

    np.savez(base_path / "embeddings.npz", **embeddings_dict)
    pd.DataFrame(metadata_rows).to_csv(base_path / "metadata.csv", index=False)
