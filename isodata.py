# isodata_4vars_jaccard.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import contingency_matrix
import os

# === ТВОЯ РЕАЛИЗАЦИЯ ISODATA (изменённая) ===

INFORMATIVE_COLS = ['Врач_ID', 'Симптомы_хэш', 'Анализы_хэш', 'Стоимость_число']
DEFAULT_K_INIT = 10 
DEFAULT_MIN_CLUSTER_SIZE = 100

# чтобы работало как надо, выбирай константы так, чтобы DEFAULT_K_INIT * DEFAULT_MIN_CLUSTER_SIZE ~ N, где N - число строк в датасете. Полученное значение кластеров и будет оптимальным. Нужно поэкспериментировать. Максимальный DEFAULT_K_INIT = 20, больше брать бессмысленно

def assign_clusters_fast(X, centers):
    # Квадрат евклидова
    dists = np.sum((X[:, None, :] - centers)**2, axis=2)
    return np.argmin(dists, axis=1)

def update_centers_fast(X, labels, k):
    return np.array([
        X[labels == i].mean(axis=0) if np.any(labels == i) else np.zeros(X.shape[1])
        for i in range(k)
    ])

def cluster_variance_fast(X, labels, centers):
    # Квадрат евклидова
    vars = []
    for i, center in enumerate(centers):
        pts = X[labels == i]
        if len(pts) > 0:
            dists = np.sum((pts - center) ** 2, axis=1)
            var = np.sum(dists)
        else:
            var = 0
        vars.append(var)
    return vars

def furthest_neighbor_distance(centers):
    # Furthest Neighbor: max расстояние между центрами
    n = len(centers)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.sum((centers[i] - centers[j])**2)  # квадрат евклидова
            dists.append(d)
    return max(dists) if dists else 0.0

def most_variant_coord_fast(X, labels, centers, idx):
    pts = X[labels == idx]
    if len(pts) > 1:
        vars = np.var(pts, axis=0)
        split_dim = np.argmax(vars)
        return split_dim, np.mean(pts[:, split_dim])
    else:
        return 0, centers[idx][0]

def split_cluster_fast(centers, idx, split_dim, split_val):
    center_a = centers[idx].copy()
    center_b = centers[idx].copy()
    delta = 0.3  # уменьшил
    center_a[split_dim] -= delta
    center_b[split_dim] += delta
    centers = np.delete(centers, idx, axis=0)
    centers = np.vstack([centers, center_a, center_b])
    return centers

def merge_clusters_fast(centers, idx_a, idx_b):
    merged = (centers[idx_a] + centers[idx_b]) / 2
    mask = np.ones(len(centers), dtype=bool)
    mask[[idx_a, idx_b]] = False
    centers = centers[mask]
    centers = np.vstack([centers, merged])
    return centers

def delete_clusters_fast(X, labels, centers, min_cluster_size):
    valid_idxs = [i for i in range(len(centers)) if np.sum(labels == i) >= min_cluster_size]
    if not valid_idxs:
        return centers, labels
    centers = centers[valid_idxs]
    mapping = {old: new for new, old in enumerate(valid_idxs)}
    labels_new = np.full_like(labels, -1)
    for old_idx in valid_idxs:
        new_idx = mapping[old_idx]
        labels_new[labels == old_idx] = new_idx
    return centers, labels_new

def isodata_clustering(
    X,
    k_init=DEFAULT_K_INIT,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    max_iter=300,
    split_delta=1.0,
    random_state=None
):
    np.random.seed(random_state if random_state is not None else 42)
    initial_idxs = np.random.choice(len(X), k_init, replace=False)
    centers = X[initial_idxs]
    converged = False
    iter_count = 0
    labels = assign_clusters_fast(X, centers)

    while not converged and iter_count < max_iter:
        iter_count += 1
        labels = assign_clusters_fast(X, centers)
        new_centers = update_centers_fast(X, labels, len(centers))
        converged = np.allclose(centers, new_centers, atol=1e-4)
        centers = new_centers

        variances = cluster_variance_fast(X, labels, centers)
        avg_variance = np.mean(variances)
        split_threshold = 1.5 * avg_variance  # увеличил

        center_dist = furthest_neighbor_distance(centers)
        merge_threshold = 0.5 * center_dist  # уменьшил — чаще сливать

        # SPLIT
        for idx, var in enumerate(variances):
            if var > split_threshold:
                split_dim, split_val = most_variant_coord_fast(X, labels, centers, idx)
                centers = split_cluster_fast(centers, idx, split_dim, split_val)
                converged = False
                break

        # MERGE
        merged = False
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sum((centers[i] - centers[j])**2)  # квадрат евклидова
                if dist < merge_threshold:
                    centers = merge_clusters_fast(centers, i, j)
                    converged = False
                    labels = assign_clusters_fast(X, centers)
                    merged = True
                    break
            if merged:
                break

        # DELETE
        centers, labels = delete_clusters_fast(X, labels, centers, min_cluster_size)
        new_labels = assign_clusters_fast(X, centers)
        if np.array_equal(labels, new_labels):
            converged = True
        labels = new_labels

    return labels, centers

# === JACCARD INDEX ===

def jaccard_index(labels_true, labels_pred):
    n_samples = len(labels_true)
    if n_samples != len(labels_pred):
        raise ValueError("Labels must have same length.")
    c = contingency_matrix(labels_true, labels_pred)
    pairs_total = n_samples * (n_samples - 1) / 2
    if pairs_total == 0:
        return 0.0

    intersection = 0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            n = c[i, j]
            intersection += n * (n - 1) / 2

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

# === ЗАПУСК ===

def run_isodata_4vars(input_path, output_dir='clustered_datasets'):
    df = pd.read_csv(input_path, sep=';', encoding='utf-8-sig')
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise ValueError("Нет числовых колонок для кластеризации.")

    missing_cols = [c for c in INFORMATIVE_COLS if c not in numeric_df.columns]
    if missing_cols:
        raise ValueError(f"Нет колонок: {missing_cols}")

    X_inf = numeric_df[INFORMATIVE_COLS]
    scaler = StandardScaler()
    X_inf_scaled = scaler.fit_transform(X_inf)

    print(f"Запуск ISODATA (4 признака) с {DEFAULT_K_INIT} кластеров...")
    labels, centers = isodata_clustering(X_inf_scaled, k_init=DEFAULT_K_INIT)

    print(f"Финальное число кластеров: {len(centers)}")

    # === JACCARD INDEX (сравнение с рандомом) ===
    random_labels = np.random.randint(0, len(centers), size=len(labels))
    jacc = jaccard_index(labels, random_labels)
    print(f"Jaccard Index (кластеры vs рандом): {jacc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    df_out = df.copy()
    df_out['cluster'] = labels
    out_path = os.path.join(output_dir, 'isodata_clustered.csv')
    df_out.to_csv(out_path, sep=';', encoding='utf-8-sig', index=False)
    print(f"Результат сохранён: {out_path}")

if __name__ == "__main__":
    run_isodata_4vars('working_data/encoded_full.csv')