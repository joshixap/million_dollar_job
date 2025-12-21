# plot_distributions.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os

def normalize_text(text):
    """Убирает лишние пробелы и приводит к нижнему регистру"""
    if isinstance(text, str):
        return re.sub(r'\s+', ' ', text.strip().lower())
    return ""

def split_and_count(series, separator=','):
    """
    Разбивает значения по разделителю и считает частоту всех элементов.
    
    Параметры:
        series : pd.Series — столбец с текстом
        separator : str — разделитель (по умолчанию запятая)
    
    Возвращает:
        Counter — словарь {значение: частота}
    """
    all_items = []
    for item in series.dropna():
        parts = [normalize_text(x.strip()) for x in str(item).split(separator)]
        all_items.extend(parts)
    return Counter(all_items)

def plot_distributions(df, columns, output_dir='plots'):
    """
    Строит распределения для указанных столбцов.
    
    Параметры:
        df : pd.DataFrame
        columns : List[str] — список названий столбцов
        output_dir : str — папка для сохранения графиков
    
    Возвращает:
        None — сохраняет графики в файлы
    """
    # Создаём папку для графиков
    os.makedirs(output_dir, exist_ok=True)
    
    for col in columns:
        if col not in df.columns:
            print(f"Колонка '{col}' не найдена")
            continue
        
        print(f"\n Анализ колонки: {col}")
        
        # Числовые — гистограмма
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f'Распределение: {col}')
            plt.xlabel(col)
            plt.ylabel('Частота')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_hist.png")
            plt.close()
            print(f"График сохранён: {output_dir}/{col}_hist.png")
        
        # Даты — преобразуем в дни от минимума, затем гистограмма
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            min_date = df[col].min()
            days_series = (df[col] - min_date).dt.days
            plt.figure(figsize=(8, 5))
            sns.histplot(days_series.dropna(), kde=True, bins=30)
            plt.title(f'Дни от первой записи: {col}')
            plt.xlabel('Дни от начала')
            plt.ylabel('Частота')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_days_hist.png")
            plt.close()
            print(f"График сохранён: {output_dir}/{col}_days_hist.png")
        
        # Категориальные и текстовые — барплот
        else:
            # Проверяем, есть ли хотя бы одна строка с запятой (мультилейбл)
            has_commas = False
            try:
                # Используем .str.contains только если тип object
                if df[col].dtype == 'object':
                    has_commas = df[col].dropna().astype(str).str.contains(',').any()
            except Exception:
                has_commas = False

            if has_commas:
                freq = split_and_count(df[col])
                top_n = 20
                if len(freq) > 0:
                    items, counts = zip(*freq.most_common(top_n))
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=list(counts), y=list(items), palette="viridis")
                    plt.title(f'Топ-{top_n} значений в {col}')
                    plt.xlabel('Частота')
                    plt.ylabel(col)
                    plt.grid(True, axis='x', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}/{col}_bar_top{top_n}.png")
                    plt.close()
                    print(f"График сохранён: {output_dir}/{col}_bar_top{top_n}.png")
                else:
                    print("Нет данных для построения")
            
            # Иначе — обычный подсчёт уникальных значений
            else:
                value_counts = df[col].value_counts().head(20)
                plt.figure(figsize=(10, 6))
                sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis")
                plt.title(f'Частота значений в {col}')
                plt.xlabel('Частота')
                plt.ylabel(col)
                plt.grid(True, axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{col}_bar.png")
                plt.close()
                print(f"График сохранён: {output_dir}/{col}_bar.png")

# Пример использования (если запускаем напрямую)
if __name__ == "__main__":
    # Загружаем полный датасет (все колонки)
    df = pd.read_csv('working_data/original_full.csv', sep=';', encoding='utf-8-sig')
    
    # Берём ВСЕ колонки
    all_columns = list(df.columns)
    print(f"Будут обработаны все колонки ({len(all_columns)} шт.): {all_columns}")
    
    # Строим графики по всем
    plot_distributions(df, all_columns, output_dir='plots')