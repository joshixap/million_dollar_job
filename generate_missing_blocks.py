# generate_missing_blocks.py

import pandas as pd
import numpy as np
from typing import List
import os
import math

def generate_dataset_with_missing_blocks(
    df: pd.DataFrame,
    missing_percent: float,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Создаёт копию датафрейма с пропусками, вставленными в виде случайных блоков
    размером от 2x2 до 4x4. Общий процент пропущенных ячеек будет не менее
    missing_percent (обычно немного больше).

    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм.
    missing_percent : float
        Целевой процент пропусков (например, 0.05 для 5%).
    random_state : int
        Для воспроизводимости.

    Возвращает:
    ----------
    pd.DataFrame
        Копия df с пропусками (NaN) в случайных блоках.
    """
    np.random.seed(random_state)
    
    n_rows, n_cols = df.shape

    # Проверка: можно ли вообще вставлять блоки 2x2?
    if n_rows < 2 or n_cols < 2:
        print("Предупреждение: датафрейм слишком мал для блоков 2x2. Пропуски не вставляются.")
        return df.copy()

    total_cells = n_rows * n_cols
    target_cells = math.ceil(total_cells * missing_percent)
    removed = 0

    df_missing = df.copy()
    removed_mask = np.zeros((n_rows, n_cols), dtype=bool)

    max_attempts = 10000
    attempts = 0

    while removed < target_cells and attempts < max_attempts:
        # Случайный размер блока: от 2 до 4
        h = np.random.randint(2, 5)  # 2, 3, 4
        w = np.random.randint(2, 5)

        # Если блок не помещается — пропускаем
        if h > n_rows or w > n_cols:
            attempts += 1
            continue

        # Максимальные допустимые стартовые позиции
        max_start_row = n_rows - h
        max_start_col = n_cols - w

        start_row = np.random.randint(0, max_start_row + 1)
        start_col = np.random.randint(0, max_start_col + 1)

        end_row = start_row + h
        end_col = start_col + w

        # Маска уже удалённых ячеек в этом блоке
        block_mask = removed_mask[start_row:end_row, start_col:end_col]
        new_removals = (~block_mask).sum()

        if new_removals == 0:
            attempts += 1
            continue

        # Устанавливаем NaN в новых ячейках
        df_missing.iloc[start_row:end_row, start_col:end_col] = df_missing.iloc[
            start_row:end_row, start_col:end_col
        ].where(block_mask)  # где mask=False → станет NaN

        # Обновляем общую маску
        removed_mask[start_row:end_row, start_col:end_col] = True
        removed += new_removals
        attempts = 0  # сброс попыток при успехе

    actual_percent = removed / total_cells * 100
    print(f"Целевой процент: {missing_percent*100:.1f}%, фактически удалено: {actual_percent:.2f}% ({removed}/{total_cells} ячеек)")

    return df_missing


def generate_all_missing_versions(
    original_path: str = 'working_data/original_full.csv',
    output_dir: str = 'missing_datasets',
    percents: List[float] = [0.05, 0.10, 0.20, 0.30],
    random_seed: int = 42
):
    """
    Генерирует несколько версий датасета с разным процентом пропусков
    и сохраняет их в CSV-файлы.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_orig = pd.read_csv(original_path, sep=';', encoding='utf-8-sig')

    for i, p in enumerate(percents):
        print(f"\n=== Генерация пропусков: {p*100:.0f}% ===")
        df_missing = generate_dataset_with_missing_blocks(
            df_orig,
            missing_percent=p,
            random_state=random_seed + i
        )
        output_file = f"{output_dir}/missing_{int(p*100)}pct.csv"
        df_missing.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';')
        print(f"Сохранено: {output_file}")