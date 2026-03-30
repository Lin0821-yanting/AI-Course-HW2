#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name(s)>
# Tatung University 14210 AI實務專題

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_benchmarks(csv_files: list[str]) -> None:
    """
    根據指定的 CSV 格式繪製圖表：
    Timestamp,Backend,PowerMode,FPS,p50_ms,p95_ms,p99_ms,gpu_memory_mb
    """
    data_frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # 直接使用你提供的欄位名稱
            # 建立標籤組合，例如 'cuda @ 25W'
            df['label'] = df['Backend'].astype(str) + " @ " + df['PowerMode'].astype(str)
            data_frames.append(df)
        except Exception as e:
            print(f"無法讀取檔案 {f}: {e}")
            continue
    
    if not data_frames:
        print("沒有可用的數據。")
        return

    combined_df = pd.concat(data_frames, ignore_index=True)

    # 圖表 1: FPS 比較長條圖 [cite: 494, 516]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(combined_df['label'], combined_df['FPS'], color='skyblue')
    plt.title("Detection FPS by Configuration")
    plt.ylabel("Detection FPS")
    plt.xticks(rotation=45)
    
    # 在長條上方標註數值 [cite: 511]
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f"{yval:.1f}", ha='center')
    
    plt.tight_layout()
    plt.savefig("chart_fps_by_configuration3.png", dpi=150) # 作業要求 150 DPI [cite: 509, 522]

    # 圖表 2: 延遲分佈群組長條圖 [cite: 495, 517]
    # 使用欄位: p50_ms, p95_ms, p99_ms
    ax = combined_df.set_index('label')[['p50_ms', 'p95_ms', 'p99_ms']].plot(
        kind='bar', figsize=(12, 6), rot=45
    )
    plt.title("Latency Distribution (ms) per Configuration")
    plt.ylabel("Latency (ms)")
    plt.legend(["p50", "p95", "p99"])
    plt.tight_layout()
    plt.savefig("chart_latency_distribution3.png", dpi=150)
    
    print("成功產生圖表：chart_fps_by_configuration.png, chart_latency_distribution.png")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方式: pdm run python visualize_benchmark.py <csv_files>")
    else:
        plot_benchmarks(sys.argv[1:])