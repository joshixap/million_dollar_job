# fill_missing.py

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
import os

MAX_ITER = 20

def iterative_fill_regression(df, random_state=42):
    """Итеративное заполнение с регрессией. X временно заполняется медианой при предсказании."""
    df_filled = df.copy()
    np.random.seed(random_state)
    
    # Считаем глобальные медианы для fallback
    global_medians = df_filled.median()
    
    for _ in range(MAX_ITER):
        if df_filled.isna().sum().sum() == 0:
            break
        
        # Заполняем каждую колонку
        for target_col in df_filled.columns:
            if df_filled[target_col].isna().any():
                feature_cols = [c for c in df_filled.columns if c != target_col]
                X = df_filled[feature_cols].copy()
                y = df_filled[target_col].copy()
                
                # Маска для обучения (где y известен)
                train_mask = y.notna()
                if train_mask.sum() < 5:
                    continue
                
                # Обучаем модель на непропущенных строках
                model = HistGradientBoostingRegressor(
                    random_state=random_state,
                    max_iter=100
                )
                model.fit(X.loc[train_mask], y.loc[train_mask])
                
                # Для предсказания: временно заполняем X медианой (чтобы не было NaN)
                X_pred = X.copy()
                for col in feature_cols:
                    X_pred[col].fillna(global_medians[col], inplace=True)
                
                # Предсказываем только пропущенные
                missing_mask = y.isna()
                if missing_mask.any():
                    y_pred = model.predict(X_pred.loc[missing_mask])
                    # Добавляем шум
                    noise_std = float(np.std(y_pred)) * 0.01
                    if not np.isnan(noise_std) and noise_std > 0:
                        noise = np.random.normal(0, noise_std, size=len(y_pred))
                        y_pred = y_pred + noise
                    y.loc[missing_mask] = y_pred
                
                df_filled[target_col] = y
    
    # Финальное заполнение остатков
    df_filled.fillna(global_medians, inplace=True)
    return df_filled

def fill_all_datasets():
    percents = [5, 10, 20, 30]
    os.makedirs('filled_datasets', exist_ok=True)
    
    for p in percents:
        print(f"\n=== Заполнение {p}% пропусков ===")
        df = pd.read_csv(f'missing_datasets/missing_{p}pct.csv', sep=';', encoding='utf-8-sig')
        print(f"Пропусков до: {df.isna().sum().sum()}")
        
        # Метод 1: Медиана
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        df_med = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        df_med = df_med.round(0).astype('Int64')
        df_med.to_csv(f'filled_datasets/filled_median_{p}pct.csv', sep=';', encoding='utf-8-sig', index=False)
        print(f"После медианы: {df_med.isna().sum().sum()}")
        
        # Метод 2: Регрессия
        df_reg = iterative_fill_regression(df.copy())
        df_reg = df_reg.round(0).astype('Int64')
        df_reg.to_csv(f'filled_datasets/filled_regression_{p}pct.csv', sep=';', encoding='utf-8-sig', index=False)
        print(f"После регрессии: {df_reg.isna().sum().sum()}")
    
    print("\n Все датасеты заполнены и округлены до целых чисел.")

if __name__ == "__main__":
    fill_all_datasets()