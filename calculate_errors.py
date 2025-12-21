# calculate_errors.py

import pandas as pd
import os
import numpy as np

def calculate_errors_for_all():
    """Рассчитывает суммарную относительную погрешность по формуле:
    ΔMj = Σ |ai - āi| / ai * 100%  (только по пропущенным ячейкам, где ai != 0)
    """
    percents = [5, 10, 20, 30]
    methods = ['median', 'regression']
    output_dir = 'working_data/error_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем оригинал
    df_original = pd.read_csv('working_data/encoded_full.csv', sep=';', encoding='utf-8-sig')
    
    results = []
    
    for p in percents:
        # Загружаем версию с пропусками (чтобы знать, где заполняли)
        df_missing = pd.read_csv(f'missing_datasets/missing_{p}pct.csv', sep=';', encoding='utf-8-sig')
        
        for method in methods:
            df_filled = pd.read_csv(f'filled_datasets/filled_{method}_{p}pct.csv', sep=';', encoding='utf-8-sig')
            
            for col in df_original.columns:
                # Маска пропущенных ячеек
                missing_mask = df_missing[col].isna()
                if not missing_mask.any():
                    continue
                
                # Истинные и восстановленные значения
                a_true = df_original.loc[missing_mask, col]
                a_filled = df_filled.loc[missing_mask, col]
                
                # Конвертируем в float для обработки <NA>
                a_true = pd.to_numeric(a_true, errors='coerce')
                a_filled = pd.to_numeric(a_filled, errors='coerce')
                
                # Исключаем строки, где истинное значение — NaN или 0
                valid = (~a_true.isna()) & (~a_filled.isna()) & (a_true != 0)
                if not valid.any():
                    total_error = 0.0
                else:
                    a_true = a_true[valid]
                    a_filled = a_filled[valid]
                    rel_errors = (a_true - a_filled).abs() / a_true
                    total_error = rel_errors.sum() * 100  # суммарная, в процентах
                
                results.append({
                    'percent': p,
                    'method': method,
                    'column': col,
                    'total_relative_error_%': total_error
                })
    
    # Сохраняем
    df_results = pd.DataFrame(results)
    output_path = f'{output_dir}/error_summary.csv'
    df_results.to_csv(output_path, index=False, sep=';', encoding='utf-8-sig')
    print(f"\n Погрешности рассчитаны и сохранены: {output_path}")
    
    # Краткий отчёт
    print("\n=== Средняя суммарная погрешность по методам ===")
    for p in percents:
        for method in methods:
            mask = (df_results['percent'] == p) & (df_results['method'] == method)
            if mask.any():
                avg_err = df_results.loc[mask, 'total_relative_error_%'].mean()
                print(f"{method:12} ({p:2}%): {avg_err:8.2f}%")

if __name__ == "__main__":
    calculate_errors_for_all()