#!/usr/bin/env python3
# Copyright (c) 2026 <Henry Tsai>
# Tatung University 14210 AI實務專題

import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_metrics(metrics_csv: str) -> None:
    """
    繪製 PR 曲線與各類別 AP 排行圖。 [cite: 515, 518]
    """
    df = pd.read_csv(metrics_csv)
    
    # 圖表 3: Precision-Recall 曲線 [cite: 500, 518]
    plt.figure(figsize=(8, 6))
    plt.plot(df['recall'], df['precision'], marker='o', label='PR Curve')
    # 標註門檻值 [cite: 502, 518]
    for i, row in df.iterrows():
        plt.annotate(f"Th:{row['threshold']}", (row['recall'], row['precision']), fontsize=8)
    
    # 突顯最佳 F1 [cite: 502, 518]
    best_f1_idx = df['f1'].idxmax()
    plt.scatter(df.loc[best_f1_idx, 'recall'], df.loc[best_f1_idx, 'precision'], color='red', s=100, label='Best F1')
    
    plt.title("Precision-Recall Curve (SSD-MobileNet v3)")# [cite: 511]
    plt.xlabel("Recall") #[cite: 511]
    plt.ylabel("Precision") #[cite: 511]
    plt.grid(True, linestyle='--', alpha=0.6) #[cite: 511]
    plt.legend()
    plt.savefig("chart_pr_curve.png", dpi=150) #[cite: 509]

    # 圖表 4: 各類別 AP@0.5 排行圖 [cite: 504, 519]
    # 自動尋找對應的 per_class CSV [cite: 506]
    per_class_csv = metrics_csv.replace("metrics_", "per_class_ap_")
    try:
        pc_df = pd.read_csv(per_class_csv).sort_values(by='ap', ascending=True)
        plt.figure(figsize=(10, 8))
        plt.barh(pc_df['class'], pc_df['ap'], color='lightgreen')
        # 繪製平均線 [cite: 519]
        plt.axvline(pc_df['ap'].mean(), color='red', linestyle='--', label=f"Mean AP: {pc_df['ap'].mean():.2f}")
        plt.title("Per-Class Average Precision (mAP@0.5)") #[cite: 511]
        plt.xlabel("Average Precision (AP)") #[cite: 511]
        plt.legend()
        plt.tight_layout()
        plt.savefig("chart_per_class_map.png", dpi=150) #[cite: 509]
    except FileNotFoundError:
        print(f"Could not find matching file: {per_class_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: pdm run python visualize_metrics.py metrics_timestamp.csv")
    else:
        plot_metrics(sys.argv[1])