# correlations_all.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chi2_contingency
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import os
import re

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def extract_price(text):
    if pd.isna(text):
        return np.nan
    match = re.search(r'[\d\s]+', str(text))
    return float(match.group().replace(' ', '')) if match else np.nan

def get_gender(name):
    if pd.isna(name):
        return "UNKNOWN"
    first = str(name).split()[0].strip().lower()
    return "жен" if first.endswith(('а', 'я')) else "муж"

def get_bin(card):
    if pd.isna(card):
        return "UNKNOWN"
    clean = re.sub(r'\D', '', str(card))
    return clean[:6] if len(clean) >= 6 else "UNKNOWN"

def normalize_multilabel(text):
    if pd.isna(text):
        return set()
    items = [re.sub(r'\s+', ' ', part.strip().lower()) for part in str(text).split(',') if part.strip()]
    return set(items)

def jaccard_similarity(s1, s2):
    union = s1 | s2
    if not union:
        return 1.0 if not s1 and not s2 else 0.0
    return len(s1 & s2) / len(union)

def cramers_v(x, y):
    crosstab = pd.crosstab(x, y)
    if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(crosstab, lambda_="log-likelihood")[0]
    n = crosstab.sum().sum()
    phi2 = chi2 / n
    r, k = crosstab.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    if min((kcorr - 1), (rcorr - 1)) <= 0:
        return 0.0
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# === ОСНОВНАЯ ФУНКЦИЯ ===

def main():
    df = pd.read_csv('working_data/original_full.csv', sep=';', encoding='utf-8-sig')
    print(f"Загружено: {df.shape}")

    # --- Числовые колонки ---
    df['Стоимость_число'] = df['Стоимость анализов'].apply(extract_price)
    df['Дата посещения врача'] = pd.to_datetime(df['Дата посещения врача'], errors='coerce')
    df['Дата получения анализов'] = pd.to_datetime(df['Дата получения анализов'], errors='coerce')
    min_visit = df['Дата посещения врача'].min()
    min_anal = df['Дата получения анализов'].min()
    df['Дни_визита'] = (df['Дата посещения врача'] - min_visit).dt.days
    df['Дни_анализов'] = (df['Дата получения анализов'] - min_anal).dt.days

    numeric_cols = ['Стоимость_число', 'Дни_визита', 'Дни_анализов']

    # --- Категориальные колонки ---
    df['Пол'] = df['ФИО'].apply(get_gender)
    df['Карта_BIN'] = df['Карта оплаты'].apply(get_bin)
    categorical_cols = ['Пол', 'Выбор врача', 'Карта_BIN']

    # --- Мультилейбл ---
    multilabel_cols = ['Симптомы', 'Анализы']

    os.makedirs('plots/correlations', exist_ok=True)

    # ============================================================== #
    # 1. ПИРСОН — все пары числовых
    # ============================================================== #
    pearson_results = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            f1, f2 = numeric_cols[i], numeric_cols[j]
            try:
                valid = (~df[f1].isna()) & (~df[f2].isna())
                if valid.sum() >= 5:
                    corr, _ = pearsonr(df[f1][valid], df[f2][valid])
                    pearson_results.append((f"{f1} vs {f2}", float(corr)))
            except:
                pass

    if pearson_results:
        pearson_results.sort(key=lambda x: abs(x[1]), reverse=True)
        names, vals = zip(*pearson_results)
        plt.figure(figsize=(10, max(4, len(names) * 0.4)))
        plt.barh(names, vals, color='steelblue')
        plt.title("Пирсон: Числовые признаки")
        plt.xlabel("Коэффициент корреляции")
        plt.tight_layout()
        plt.savefig("plots/correlations/pearson.png", dpi=150)
        plt.close()
        pd.DataFrame(pearson_results, columns=['pair', 'value']).to_csv(
            'plots/correlations/pearson.csv', sep=';', encoding='utf-8-sig', index=False
        )
        print(f"Пирсон: {len(pearson_results)} пар")

    # ============================================================== #
    # 2. КРАМЕР V — все пары категориальных
    # ============================================================== #
    cramer_results = []
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            f1, f2 = categorical_cols[i], categorical_cols[j]
            try:
                x = df[f1].fillna("MISSING").astype(str)
                y = df[f2].fillna("MISSING").astype(str)
                valid = (x != "MISSING") & (y != "MISSING")
                if valid.sum() >= 10:
                    v = cramers_v(x[valid], y[valid])
                    cramer_results.append((f"{f1} vs {f2}", float(v)))
            except:
                pass

    # Дополнительно: категория ↔ стоимость (через MI, но пока не сюда)
    if cramer_results:
        cramer_results.sort(key=lambda x: x[1], reverse=True)
        names, vals = zip(*cramer_results)
        plt.figure(figsize=(10, max(4, len(names) * 0.4)))
        plt.barh(names, vals, color='seagreen')
        plt.title("Крамер V: Категориальные признаки")
        plt.xlabel("V")
        plt.tight_layout()
        plt.savefig("plots/correlations/cramer.png", dpi=150)
        plt.close()
        pd.DataFrame(cramer_results, columns=['pair', 'value']).to_csv(
            'plots/correlations/cramer.csv', sep=';', encoding='utf-8-sig', index=False
        )
        print(f"Крамер: {len(cramer_results)} пар")

    # ============================================================== #
    # 3. ЖАККАРД — мультилейбл vs мультилейбл
    # ============================================================== #
    jaccard_results = []
    if len(multilabel_cols) == 2:
        f1, f2 = multilabel_cols
        sets1 = df[f1].apply(normalize_multilabel)
        sets2 = df[f2].apply(normalize_multilabel)
        jaccards = []
        valid_count = 0
        for s1, s2 in zip(sets1, sets2):
            # Считаем, если хотя бы один набор не пустой
            if len(s1) > 0 or len(s2) > 0:
                jaccards.append(jaccard_similarity(s1, s2))
                valid_count += 1

        if jaccards:
            mean_j = np.mean(jaccards)
            jaccard_results.append((f"{f1} vs {f2}", float(mean_j)))
            print(f"Жаккард: обработано {valid_count} непустых строк")
        else:
            print("Жаккард: нет данных для расчёта (все строки пустые)")
    else:
        print("Жаккард: не найдены мультилейбл-колонки")

    if jaccard_results:
        names, vals = zip(*jaccard_results)
        plt.figure(figsize=(6, 2.5))
        plt.barh(names, vals, color='orange')
        plt.title("Жаккард: Симптомы vs Анализы")
        plt.xlabel("Средний коэффициент Жаккарда")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig("plots/correlations/jaccard.png", dpi=150)
        plt.close()
        pd.DataFrame(jaccard_results, columns=['pair', 'value']).to_csv(
            'plots/correlations/jaccard.csv', sep=';', encoding='utf-8-sig', index=False
        )
        print(f"Жаккард: {len(jaccard_results)} пар")
    else:
        print("Жаккард: график и CSV не сохранены (недостаточно данных)")

    # ============================================================== #
    # 4. ВЗАИМНАЯ ИНФОРМАЦИЯ — категория/мультилейбл vs СТОИМОСТЬ
    # ============================================================== #
    mi_results = []
    all_cat_ml = categorical_cols + multilabel_cols  # все нечисловые, кроме ID

    for col in all_cat_ml:
        try:
            x = df[col].fillna("MISSING").astype(str)
            y = df['Стоимость_число']
            valid = (~x.isna()) & (~y.isna()) & (y > 0)
            if valid.sum() >= 10:
                x_enc = LabelEncoder().fit_transform(x[valid])
                mi = mutual_info_regression(
                    x_enc.reshape(-1, 1), y[valid], random_state=42, n_neighbors=min(3, len(x_enc)-1)
                )[0]
                mi_results.append((f"{col} vs Стоимость_число", float(mi)))
        except:
            pass

    if mi_results:
        mi_results.sort(key=lambda x: x[1], reverse=True)
        names, vals = zip(*mi_results)
        plt.figure(figsize=(10, max(4, len(names) * 0.4)))
        plt.barh(names, vals, color='purple')
        plt.title("Взаимная информация: Признаки vs Стоимость")
        plt.xlabel("MI")
        plt.tight_layout()
        plt.savefig("plots/correlations/mutual_info.png", dpi=150)
        plt.close()
        pd.DataFrame(mi_results, columns=['pair', 'value']).to_csv(
            'plots/correlations/mutual_info.csv', sep=';', encoding='utf-8-sig', index=False
        )
        print(f"Взаимная информация: {len(mi_results)} пар")

    print("\n Готово. Все графики и CSV в: plots/correlations/")
    print("Файлы:")
    print(" - pearson.png/csv")
    print(" - cramer.png/csv")
    print(" - jaccard.png/csv")
    print(" - mutual_info.png/csv")

if __name__ == "__main__":
    main()