# fuzzy_isodata_jaccard.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import contingency_matrix
import os

def fuzzy_kmeans(X, n_clusters=3, m=2, max_iter=100, tol=1e-4, random_state=42):
    """Fuzzy KMeans с возвращением меток и матрицы принадлежности."""
    np.random.seed(random_state)
    n_samples, n_features = X.shape

    u = np.random.rand(n_samples, n_clusters)
    u = u / u.sum(axis=1, keepdims=True)

    for iteration in range(max_iter):
        centers = np.zeros((n_clusters, n_features))
        for c in range(n_clusters):
            weights = u[:, c] ** m
            centers[c] = (weights[:, np.newaxis] * X).sum(axis=0) / weights.sum()

        new_u = np.zeros_like(u)
        for i in range(n_samples):
            for c in range(n_clusters):
                dist = np.linalg.norm(X[i] - centers[c])
                if dist == 0:
                    new_u[i, c] = 1
                    continue
                inv_dist = 1.0 / np.linalg.norm(X[i] - centers, axis=1)
                new_u[i, c] = 1.0 / np.sum((dist * inv_dist) ** (2.0 / (m - 1)))

        if np.linalg.norm(new_u - u) < tol:
            break
        u = new_u

    labels = np.argmax(u, axis=1)
    return labels, u, centers

def jaccard_index_from_labels(labels_true, labels_pred):
    """Считает индекс Жаккара между двумя разбиениями (не ground-truth)."""
    n_samples = len(labels_true)
    if n_samples != len(labels_pred):
        raise ValueError("Labels must have same length.")

    c = contingency_matrix(labels_true, labels_pred)
    # Считаем пары: сколько точек в одной паре кластеров
    pairs_total = n_samples * (n_samples - 1) / 2
    if pairs_total == 0:
        return 0.0

    # Число пар в одном кластере в обоих разбиениях
    intersection = 0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            n = c[i, j]
            intersection += n * (n - 1) / 2

    # Объединение: сумма для каждого кластера в true и pred
    union = 0
    for i in range(c.shape[0]):
        n = c[i, :].sum()
        union += n * (n - 1) / 2
    for j in range(c.shape[1]):
        n = c[:, j].sum()
        union += n * (n - 1) / 2
    union -= intersection

    if union == 0:
        return 0.0

    return intersection / union

def fuzzy_isodata_jaccard_analysis(input_path, output_dir='clustered_datasets', n_start=2, n_end=20):
    """Анализ Fuzzy ISODATA с Jaccard Index."""
    df = pd.read_csv(input_path, sep=';', encoding='utf-8-sig')
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("Нет числовых колонок для кластеризации.")

    scaler = StandardScaler()
    X = scaler.fit_transform(numeric_df)
    print("Данные нормированы (StandardScaler).")

    results = []
    os.makedirs(output_dir, exist_ok=True)

    prev_labels = None

    for n in range(n_start, n_end + 1):
        print(f"Кластеризация для N = {n}...")
        labels, u, centers = fuzzy_kmeans(X, n_clusters=n)

        # Сохраняем датасет с кластерами
        df_out = df.copy()
        df_out['cluster'] = labels
        out_path = os.path.join(output_dir, f'clustered_N{n}.csv')
        df_out.to_csv(out_path, sep=';', encoding='utf-8-sig', index=False)

        # Сравниваем с предыдущим разбиением
        if prev_labels is not None:
            jacc = jaccard_index_from_labels(prev_labels, labels)
        else:
            jacc = 0.0  # для N=2 нет предыдущего

        results.append({'n_clusters': n, 'jaccard_index': jacc})
        prev_labels = labels.copy()

    # График
    df_results = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['n_clusters'], df_results['jaccard_index'], marker='o', linestyle='-', color='steelblue')
    plt.title('Fuzzy ISODATA: Jaccard Index vs N кластеров')
    plt.xlabel('Число кластеров (N)')
    plt.ylabel('Индекс Жаккара')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jaccard_vs_n.png'), dpi=150)
    plt.close()

    print(f"Результаты сохранены в: {output_dir}")
    print(df_results)

if __name__ == "__main__":
    fuzzy_isodata_jaccard_analysis('working_data/encoded_full.csv')