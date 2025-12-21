# input.py
import pandas as pd
import os

def load_and_clean_dataset(input_path='input_data/clinic_dataset.xlsx',
                          output_path='working_data/original_full.csv'):
    """
    Читает исходный Excel-файл и сохраняет его в CSV без удаления колонок.
    Никакие колонки не удаляются — сохраняется полная структура.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Загрузка исходного датасета...")
    df = pd.read_excel(input_path, sep=';', encoding='utf-8-sig')
    print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Колонки: {list(df.columns)}")

    df.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';')
    print(f"Полный датасет сохранён: {output_path}")
    return df

if __name__ == "__main__":
    original_df = load_and_clean_dataset()