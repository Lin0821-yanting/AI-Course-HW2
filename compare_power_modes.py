#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name(s)>
# Tatung University — I4210 AI實務專題

import csv
import argparse
import re
import sys
import os
from typing import Dict, Optional

# ANSI 顏色代碼：不需額外套件即可在終端機顯示顏色
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

class BenchmarkData:
    """解析並儲存基準測試 CSV 資料的類別。"""

    def __init__(self, filepath: str):
        self.filepath = os.path.basename(filepath)
        self.power_mode = "Unknown"
        self.fps = 0.0
        self.p50 = 0.0
        self.p95 = 0.0
        self.p99 = 0.0
        self.mem = 0.0
        self.wattage: Optional[float] = None
        self._load_data(filepath)

    def _load_data(self, path: str):
        try:
            with open(path, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                row = next(reader)
                
                # 支援多種欄位命名格式 [cite: 346, 357]
                self.power_mode = row.get('PowerMode') or row.get('Power Mode') or "Unknown"
                self.fps = float(row.get('FPS') or 0.0)
                self.p50 = float(row.get('p50_ms') or 0.0)
                self.p95 = float(row.get('p95_ms') or 0.0)
                self.p99 = float(row.get('p99_ms') or 0.0)
                self.mem = float(row.get('gpu_memory_mb') or 0.0)
                
                # 從功率模式字串中提取數字 (例如 "15W" -> 15.0) 
                match = re.search(r'(\d+)', self.power_mode)
                if match:
                    self.wattage = float(match.group(1))
        except Exception as e:
            print(f"Error reading {path}: {e}")
            sys.exit(1)

    @property
    def fps_per_watt(self) -> float:
        """計算 FPS-per-Watt 效率 。"""
        return self.fps / self.wattage if self.wattage else 0.0

def format_change(base: float, target: float, lower_is_better: bool = False) -> str:
    """計算百分比變化並套用顏色 。"""
    if base == 0: return "N/A"
    change = ((target - base) / base) * 100
    
    # 決定顏色邏輯
    is_improvement = (change < 0) if lower_is_better else (change > 0)
    color = GREEN if is_improvement else RED
    if abs(change) < 0.01: color = RESET # 無明顯變化
    
    return f"{color}{change:+.1f}%{RESET}"

def run_comparison(csv1: str, csv2: str):
    d1 = BenchmarkData(csv1)
    d2 = BenchmarkData(csv2)

    # 列印檔名與模式資訊 
    print(f"\n{BOLD}Comparing:{RESET} {CYAN}{d1.power_mode}{RESET} ({d1.filepath})")
    print(f"      {BOLD}vs:{RESET} {CYAN}{d2.power_mode}{RESET} ({d2.filepath})\n")

    # 定義表格標題與欄位寬度
    headers = ["Metric", d1.power_mode, d2.power_mode, "Change"]
    w = [22, 12, 12, 12] # 欄位寬度
    
    # 列印表頭 (Markdown 風格) 
    header_line = f"| {' | '.join(h.ljust(w[i]) for i, h in enumerate(headers))} |"
    sep_line = f"| {' | '.join('-' * w[i] for i in range(len(w)))} |"
    print(header_line)
    print(sep_line)

    # 定義要顯示的指標 (名稱, 屬性, 是否越低越好)
    metrics = [
        ("Detection FPS", d1.fps, d2.fps, False),
        ("Latency p50 (ms)", d1.p50, d2.p50, True),
        ("Latency p95 (ms)", d1.p95, d2.p95, True),
        ("Latency p99 (ms)", d1.p99, d2.p99, True),
        ("GPU Memory (MB)", d1.mem, d2.mem, True),
        ("FPS per Watt", d1.fps_per_watt, d2.fps_per_watt, False),
    ]

    for label, v1, v2, low_better in metrics:
        change_str = format_change(v1, v2, low_better)
        # ANSI 顏色會增加字串長度，需處理對齊問題
        v1_str = f"{v1:.2f}"
        v2_str = f"{v2:.2f}"
        print(f"| {label.ljust(w[0])} | {v1_str.ljust(w[1])} | {v2_str.ljust(w[2])} | {change_str.ljust(w[3] + 9)} |") # 9 是補償 ANSI 長度

    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jetson Power Mode Benchmark Comparator")
    parser.add_argument("csv1", help="Path to base benchmark CSV")
    parser.add_argument("csv2", help="Path to target benchmark CSV")
    args = parser.parse_args()
    run_comparison(args.csv1, args.csv2)