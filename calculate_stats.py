# calculate_stats.py
from typing import Dict, List, Optional, Any
import pandas as pd

def calculate_stats(
    df: pd.DataFrame,
    columns: List[str]
) -> Dict[str, Dict[str, Optional[Any]]]:
    """
    Считает среднее, медиану и моду для указанных столбцов датафрейма.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный датафрейм с медицинскими данными.
    columns : List[str]
        Список названий столбцов, для которых нужно посчитать статистики.
    
    Возвращает:
    ----------
    Dict[str, Dict[str, Optional[Any]]]
        Словарь, где:
        - ключ — название колонки (str),
        - значение — словарь вида:
            {
                'среднее': float или None (если нечисловой тип),
                'медиана': float или None (если нечисловой тип),
                'мода': Any (чаще str или float) или None (если данных нет)
            }
    
    Примечания:
    ----------
    - Пропущенные значения (NaN) игнорируются.
    - Для нечисловых колонок 'среднее' и 'медиана' всегда None.
    - Если колонка пустая (все NaN), все статистики будут None.
    """
    results: Dict[str, Dict[str, Optional[Any]]] = {}
    
    for col in columns:
        if col not in df.columns:
            print(f"Внимание: колонка '{col}' отсутствует в датафрейме")
            continue
        
        # Убираем NaN
        series: pd.Series = df[col].dropna()
        stats: Dict[str, Optional[Any]] = {
            'среднее': None,
            'медиана': None,
            'мода': None
        }
        
        # Проверяем, числовой ли тип (включая int, float)
        if pd.api.types.is_numeric_dtype(series):
            stats['среднее'] = float(series.mean())
            stats['медиана'] = float(series.median())
        
        # Мода — для любого типа (берём первую, если несколько)
        if not series.empty:
            mode_vals: pd.Series = series.mode()
            stats['мода'] = mode_vals.iloc[0] if not mode_vals.empty else None
        
        results[col] = stats
    
    return results