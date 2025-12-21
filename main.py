# main.py

import pandas as pd
import json
import os
from calculate_stats import calculate_stats
from plot_distributions import plot_distributions
from generate_missing_blocks import generate_all_missing_versions
from correlations_all import main as run_correlations
from encode_full_dataset import main as encode_dataset
from fill_missing import fill_all_datasets
from calculate_errors import calculate_errors_for_all  # ← НОВЫЙ ИМПОРТ
from isodata import run_isodata_4vars


def save_stats_for_missing_datasets():
    """Сохраняет статистики для датасетов С ПРОПУСКАМИ (для оценки смещения от пропусков)."""
    percents = [5, 10, 20, 30]
    stats_dir = 'working_data/missing_stats'
    os.makedirs(stats_dir, exist_ok=True)

    for p in percents:
        path = f'missing_datasets/missing_{p}pct.csv'
        if not os.path.exists(path):
            print(f"Пропущено: {path} не найден")
            continue

        df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
        stats = calculate_stats(df, df.columns.tolist())

        output_path = f'{stats_dir}/stats_missing_{p}pct.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=4, default=str)
        print(f"Статистика по пропускам сохранена: {output_path}")


def save_stats_for_filled_datasets():
    """Сохраняет статистики для всех восстановленных датасетов."""
    percents = [5, 10, 20, 30]
    methods = ['median', 'regression']
    stats_dir = 'working_data/filled_stats'
    os.makedirs(stats_dir, exist_ok=True)

    for p in percents:
        for method in methods:
            path = f'filled_datasets/filled_{method}_{p}pct.csv'
            if not os.path.exists(path):
                print(f"Пропущено: {path} не найден")
                continue

            df = pd.read_csv(path, sep=';', encoding='utf-8-sig')
            stats = calculate_stats(df, df.columns.tolist())

            output_path = f'{stats_dir}/stats_filled_{method}_{p}pct.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=4, default=str)
            print(f"Статистика по восстановленным сохранена: {output_path}")


def main():
    """
    Полный конвейер лабораторной работы:

    Этап 1: Анализ исходного датасета (для отчёта)
    Этап 2: Оцифровка → числовой датасет
    Этап 3: Генерация пропусков
    Этап 4: Статистика по версиям с пропусками (для оценки смещения!)
    Этап 5: Заполнение пропусков двумя методами
    Этап 6: Статистика по восстановленным данным
    Этап 7: Расчёт погрешностей восстановления
    Этап 8: Кластеризация ISODATA на оригинальном датасете
    """

    # === Пути ===
    input_excel = 'input_data/clinic_dataset.xlsx'
    original_csv = 'working_data/original_full.csv'
    encoded_csv = 'working_data/encoded_full.csv'
    stats_json = 'working_data/original_stats.json'
    plots_dir = 'plots'
    missing_dir = 'missing_datasets'
    filled_dir = 'filled_datasets'

    # === Создание папок ===
    os.makedirs('working_data', exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(missing_dir, exist_ok=True)
    os.makedirs(filled_dir, exist_ok=True)

    # === Шаг 1: Загрузка исходного датасета ===
    print("=== Шаг 1: Загрузка исходного датасета ===")
    if not os.path.exists(input_excel):
        print(f"Ошибка: файл не найден — {input_excel}")
        return

    df_original = pd.read_excel(input_excel)
    print(f"Загружено: {df_original.shape[0]} строк, {df_original.shape[1]} колонок")
    df_original.to_csv(original_csv, index=False, encoding='utf-8-sig', sep=';')
    print(f"Оригинал сохранён: {original_csv}")

    # === Шаг 2: Статистика по оригиналу ===
    print("\n=== Шаг 2: Расчёт статистики по оригиналу ===")
    all_columns = list(df_original.columns)
    stats = calculate_stats(df_original, all_columns)
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4, default=str)
    print(f"Статистика сохранена: {stats_json}")

    # === Шаг 3: Графики по оригиналу ===
    print("\n=== Шаг 3: Построение графиков распределений (оригинал) ===")
    plot_distributions(df_original, all_columns, output_dir=plots_dir)
    print(f"Графики сохранены в: {plots_dir}")

    # === Шаг 4: Корреляции по оригиналу ===
    print("\n=== Шаг 4: Анализ корреляций (Пирсон, Крамер, Жаккард, MI) ===")
    run_correlations()
    print("Корреляции сохранены в: plots/correlations/")

    # === Шаг 5: Оцифровка всех колонок ===
    print("\n=== Шаг 5: Оцифровка всех 10 колонок в числовой формат ===")
    encode_dataset()
    print(f"Оцифрованный датасет сохранён: {encoded_csv}")

    # === Шаг 6: Генерация пропусков в оцифрованном датасете ===
    print("\n=== Шаг 6: Генерация блоковых пропусков в оцифрованном датасете ===")
    generate_all_missing_versions(
        original_path=encoded_csv,
        output_dir=missing_dir,
        percents=[0.05, 0.10, 0.20, 0.30],
        random_seed=42
    )
    print(f"Созданы файлы с пропусками в: {missing_dir}")

    # === Шаг 6.5: Статистики по версиям с пропусками ===
    print("\n=== Шаг 6.5: Расчёт статистик по версиям с пропусками (для оценки смещения) ===")
    save_stats_for_missing_datasets()
    print("Статистики по данным с пропусками сохранены.")

    # === Шаг 7: Заполнение пропусков двумя методами ===
    print("\n=== Шаг 7: Заполнение пропусков двумя методами ===")
    fill_all_datasets()
    print(f"Заполненные датасеты сохранены в: {filled_dir}")

    # === Шаг 8: Статистики по восстановленным датасетам ===
    print("\n=== Шаг 8: Расчёт статистик по восстановленным датасетам ===")
    save_stats_for_filled_datasets()
    print("Статистики по всем восстановленным данным сохранены.")

    # === ШАГ 9: РАСЧЁТ ПОГРЕШНОСТЕЙ ВОССТАНОВЛЕНИЯ ===
    print("\n=== Шаг 9: Расчёт суммарных относительных погрешностей ===")
    calculate_errors_for_all()
    print("Погрешности сохранены в: working_data/error_analysis/error_summary.csv")

    # === ШАГ 10: КЛАСТЕРИЗАЦИЯ ISODATA НА ОРИГИНАЛЬНОМ ОЦИФРОВАННОМ ДАТАСЕТЕ ===
    print("\n=== Шаг 10: Кластеризация ISODATA на оригинальном оцифрованном датасете ===")
    run_isodata_4vars('working_data/encoded_full.csv')
    print("Кластеризованный датасет сохранён: clustered_datasets/isodata_clustered.csv")

    print("\n" + "="*60)
    print(" ВСЕ ЭТАПЫ ВЫПОЛНЕНЫ.")
    print("\nСоздано:")
    print("- Оригинал, его статистика и графики")
    print("- Оцифрованный датасет")
    print("- Версии с пропусками + статистики")
    print("- Версии с заполненными пропусками + статистики")
    print("- Файл погрешностей: error_summary.csv")
    print("- Кластеризованный датасет: isodata_clustered.csv")
    print("\nТеперь можно:")
    print(" Анализировать кластеры (массовые vs сложные случаи)")
    print(" Оценивать смещение от пропусков")
    print(" Сравнивать методы заполнения")
    print(" Делать выводы по качеству восстановления и структуре данных")
    print("="*60)


if __name__ == "__main__":
    main()