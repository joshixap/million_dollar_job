# find_best_isodata_params.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from isodata import (
    assign_clusters_fast,
    update_centers_fast,
    cluster_variance_fast,
    furthest_neighbor_distance,
    most_variant_coord_fast,
    split_cluster_fast,
    merge_clusters_fast,
    delete_clusters_fast,
    jaccard_index
)

def isodata_clustering(
    X,
    k_init=20,
    min_cluster_size=50,
    max_iter=300,
    split_delta=1.0,
    random_state=42
):
    np.random.seed(random_state)
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
        split_threshold = 1.5 * avg_variance

        center_dist = furthest_neighbor_distance(centers)
        merge_threshold = 0.5 * center_dist

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
                dist = np.sum((centers[i] - centers[j])**2)
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

INFORMATIVE_COLS = ['Врач_ID', 'Симптомы_хэш', 'Анализы_хэш', 'Стоимость_число']

def find_best_params(input_path):
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

    n_samples = len(X_inf)

    best_k = 1
    best_min_size = 1
    best_jacc = -1.0

    results = []

    print("Поиск лучших параметров ISODATA...")

    for k in range(1, 21):
        min_size = int(n_samples / k)  # округляем вниз
        if min_size < 1:
            min_size = 1

        print(f"Тест: K={k}, MinSize={min_size}...", end=" ")

        try:
            labels, centers = isodata_clustering(X_inf_scaled, k_init=k, min_cluster_size=min_size, random_state=42)
            random_labels = np.random.randint(0, len(centers), size=len(labels))
            jacc = jaccard_index(labels, random_labels)
            print(f"Jaccard = {jacc:.4f}")

            results.append({'k_init': k, 'min_cluster_size': min_size, 'jaccard_index': jacc})

            if jacc > best_jacc:
                best_jacc = jacc
                best_k = k
                best_min_size = min_size

        except Exception as e:
            print(f"Ошибка: {e}")
            results.append({'k_init': k, 'min_cluster_size': min_size, 'jaccard_index': np.nan})
            continue

    # === ГРАФИК ===
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['k_init'], df_results['jaccard_index'], marker='o', linestyle='-', color='steelblue')
    plt.title('ISODATA: Jaccard Index vs K_INIT')
    plt.xlabel('K_INIT')
    plt.ylabel('Jaccard Index')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('isodata_jaccard_plot.png', dpi=150)
    plt.show()

    return best_k, best_min_size

if __name__ == "__main__":
    find_best_params('working_data/encoded_full.csv')