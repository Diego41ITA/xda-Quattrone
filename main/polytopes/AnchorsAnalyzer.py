import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
import sys

# =========================
# Parsing intervalli e distanza punto->box
# =========================
def parse_interval(s):
    s = str(s).strip("()")
    low, high, low_inc, high_inc = [x.strip() for x in s.split(",")]
    low = -np.inf if low == "-inf" else float(low)
    high = np.inf if high == "inf" else float(high)
    low_inc = low_inc.lower() == "true"
    high_inc = high_inc.lower() == "true"
    return low, high, low_inc, high_inc

def point_to_interval_distance(x, interval):
    low, high, low_inc, high_inc = interval
    if x < low or (x == low and not low_inc):
        return low - x
    elif x > high or (x == high and not high_inc):
        return x - high
    else:
        return 0.0

def point_to_box_distance(point_row, box_row, feature_cols):
    sq_sum = 0.0
    for col in feature_cols:
        interval = parse_interval(box_row[col])
        d = point_to_interval_distance(point_row[col], interval)
        sq_sum += d ** 2
    return math.sqrt(sq_sum)

# =========================
# Calcolo distanza media
# =========================
def compute_all_distances(df_points, df_boxes, feature_cols):
    results = {2: [], 3: [], 4: []}

    for _, point in df_points.iterrows():
        dists = {2: [], 3: [], 4: []}
        for _, box in df_boxes.iterrows():
            n_req = box["n_satisfied_reqs"]
            if n_req in dists:
                dist = point_to_box_distance(point, box, feature_cols)
                dists[n_req].append(dist)
        for k in dists:
            if dists[k]:
                results[k].append(np.mean(dists[k]))  # distanza media per quel punto
    return results  # restituiamo tutti i valori, non solo la media

# =========================
# Funzione boxplot
# =========================
def plot_avg_distances_box(avg_dists, filename="distanza_media_boxplot.png", title_suffix=""):
    keys = sorted(avg_dists.keys())
    data = [avg_dists[k] for k in keys]

    plt.figure()
    plt.boxplot(data, labels=[str(k) for k in keys], showmeans=True)
    plt.xlabel("Numero requisiti soddisfatti")
    plt.ylabel("Distanza euclidea media")
    plt.title(f"Distanza media dai box vs requisiti soddisfatti {title_suffix}")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Boxplot salvato in: {filename}")

# =========================
# Trova i vicini point->box e restituisce anche i conteggi
# =========================
def find_neighbors_point_to_box(df_points, df_boxes, feature_cols,
                                n_samples=10, top_k=5, output_file="neighbors_summary.csv"):
    rows = []

    for idx, point in df_points.iterrows():
        distances = []
        for i, box in df_boxes.iterrows():
            d = point_to_box_distance(point, box, feature_cols)
            distances.append((i, d))

        # Ordina e prendi top_k vicini
        distances.sort(key=lambda x: x[1])
        top_neighbors = distances[:top_k]

        # Conta vicini per numero requisiti
        counts = {2:0, 3:0, 4:0}
        for i, _ in top_neighbors:
            n_req = df_boxes.iloc[i]["n_satisfied_reqs"]
            if n_req in counts:
                counts[n_req] += 1

        # Aggiungi riga da salvare
        row = {
            "point_index": idx,
            "top_k": top_k,
            "neighbors_2": counts[2],
            "neighbors_3": counts[3],
            "neighbors_4": counts[4]
        }
        rows.append(row)

    # Salva in CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_file, index=False)
    print(f"Dati dei vicini salvati in: {output_file}")

    return df_out  # ritorniamo il dataframe per il boxplot

# =========================
# Boxplot vicini
# =========================
def plot_neighbors_box(df_neighbors, filename="neighbors_boxplot.png"):
    data = [df_neighbors[f"neighbors_{k}"] for k in [2,3,4]]

    plt.figure()
    plt.boxplot(data, labels=["2", "3", "4"], showmeans=True)
    plt.xlabel("Numero requisiti soddisfatti")
    plt.ylabel("Conteggio vicini top_k")
    plt.title("Distribuzione dei vicini top_k per numero requisiti soddisfatti")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Boxplot vicini salvato in: {filename}")

# =========================
# Definizione feature
# =========================
featureNames = ["cruise speed",
                "image resolution",
                "illuminance",
                "controls responsiveness",
                "power",
                "smoke intensity",
                "obstacle size",
                "obstacle distance",
                "firm obstacle"]

controllableFeaturesNames = featureNames[0:3]
externalFeaturesNames = featureNames[3:7]

# =========================
# Main
# =========================
if __name__ == '__main__':
    os.chdir(sys.path[0])

    df_initial = pd.read_csv("../datasets/balanced_dataset.csv")
    df_compare = pd.read_csv("anchors_explanations.csv")

    # Pulizia colonne e tipi
    df_initial.columns = df_initial.columns.str.strip()
    df_compare.columns = df_compare.columns.str.strip()
    df_initial["firm obstacle"] = df_initial["firm obstacle"].astype(float)
    df_compare["n_satisfied_reqs"] = df_compare["n_satisfied_reqs"].astype(int)

    # ===== Parte 1: distanza media (boxplot)
    avg_all = compute_all_distances(df_initial, df_compare, featureNames)
    plot_avg_distances_box(avg_all, filename="distanza_media_boxplot_tutte.png", title_suffix="(tutte le feature)")

    avg_cont = compute_all_distances(df_initial, df_compare, controllableFeaturesNames)
    plot_avg_distances_box(avg_cont, filename="distanza_media_boxplot_controllabili.png", title_suffix="(solo controllabili)")

    # ===== Parte 2: vicini point->box (boxplot)
    df_neighbors = find_neighbors_point_to_box(df_initial, df_compare, featureNames,
                                               top_k=15,
                                               output_file="neighbors_summary.csv")
    plot_neighbors_box(df_neighbors, filename="neighbors_boxplot.png")
