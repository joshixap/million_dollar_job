# feature_spread_selection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_feature_spread(input_path, output_dir='plots/feature_spread'):
    """Анализирует разброс признаков (дисперсия) и строит график."""
    df = pd.read_csv(input_path, sep=';', encoding='utf-8-sig')
    
    # Выбираем числовые колонки
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("Нет числовых колонок для анализа.")
    
    # Считаем дисперсию
    variances = numeric_df.var().sort_values(ascending=False)
    
    # Сохраняем
    os.makedirs(output_dir, exist_ok=True)
    variances.to_csv(os.path.join(output_dir, 'feature_variances.csv'), sep=';', encoding='utf-8-sig')
    
    # График
    plt.figure(figsize=(12, 6))
    plt.bar(variances.index, variances.values, color='steelblue')
    plt.title('Разброс (дисперсия) признаков')
    plt.ylabel('Дисперсия')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_spread.png'), dpi=150)
    plt.close()
    
    print(f"Анализ разброса сохранён в: {output_dir}")
    print("Информативные признаки (по дисперсии):")
    print(variances.head(10))

if __name__ == "__main__":
    analyze_feature_spread('working_data/encoded_full.csv')