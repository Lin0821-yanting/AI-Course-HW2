#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name(s)>
# Tatung University — I4210 AI實務專題

import os
import cv2
import time
import csv
import argparse
import psutil
import numpy as np
from datetime import datetime

# 導入自定義類別
from camera import Camera
from detector import Detector

try:
    from jtop import jtop
except ImportError:
    print("Warning: jtop library not found. Power mode detection will be disabled.")
    jtop = None

class BenchmarkRunner:
    """
    量測框架類別：協調 Camera 與 Detector 進行效能評測。
    """

    def __init__(self, pb_path: str, pbtxt_path: str, backend: str, warmups: int = 50, runs: int = 500):
        self.pb_path = pb_path
        self.pbtxt_path = pbtxt_path
        self.backend = backend
        self.warmups = warmups
        self.runs = runs
        
        # 初始化偵測器
        self.detector = Detector(
            pb_file=self.pb_path, 
            pbtxt_file=self.pbtxt_path, 
            backend=self.backend
        )
        
        # 初始化相機 (使用原生模式 1280x720)
        self.camera = Camera(width=1280, height=720, fps=60)

    def _get_power_mode(self) -> str:
        """偵測當前 Jetson 功率模式。"""
        if jtop is None:
            return "Unknown"
        with jtop() as j:
            if j.ok:
                return j.nvpmodel.name
        return "N/A"

    def _get_system_memory_mb(self) -> float:
        """取得系統記憶體使用量 (Jetson UMA 架構)。"""
        # 參考作業提示： nvidia-smi 在 Jetson 回傳 N/A，改採 /proc/meminfo 或 psutil
        return psutil.virtual_memory().used / (1024 * 1024) 

    def _save_results(self, fw_latencies: list, e2e_latencies: list, power_mode: str, peak_mem: float) -> None:
        """計算統計數據並存檔。"""
        p50 = np.percentile(fw_latencies, 50)
        p95 = np.percentile(fw_latencies, 95)
        p99 = np.percentile(fw_latencies, 99) 
        
        avg_e2e = np.mean(e2e_latencies)
        fps = 1000.0 / avg_e2e if avg_e2e > 0 else 0.0 

        print(f"\n--- {self.backend} 結果分析 ---")
        print(f"FPS: {fps:.2f} | P50: {p50:.2f}ms | P99: {p99:.2f}ms") 

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{self.backend}_{timestamp}.csv"

        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Backend", "PowerMode", "FPS", "p50_ms", "p95_ms", "p99_ms", "gpu_memory_mb"])
            writer.writerow([timestamp, self.backend, power_mode, f"{fps:.2f}", f"{p50:.2f}", f"{p95:.2f}", f"{p99:.2f}", f"{peak_mem:.2f}"]) 
        
        print(f"結果已存至: {filename}")

    def run(self) -> None:
        """執行基準測試主迴圈。"""
        power_mode = self._get_power_mode()
        fw_latencies = []
        e2e_latencies = []
        peak_memory = 0.0
        total_iters = self.warmups + self.runs

        print(f"啟動測試: 後端={self.backend}, 模式={power_mode}")
        
        try:
            for i in range(total_iters):
                # 【端到端起點】
                t_e2e_start = time.perf_counter()

                frame = self.camera.read()
                if frame is None:
                    break
                
                h, w = frame.shape[:2]

                # 1. 前處理 (不計入純推理延遲)
                blob = self.detector.preprocess(frame) 

                # 2. 推理運算 (核心計時對象)
                t_fw_start = time.perf_counter()
                raw_dets = self.detector.forward(blob) 
                t_fw_end = time.perf_counter()
                
                # 3. 後處理 (解析結果)
                _ = self.detector.postprocessing(raw_dets, w, h)

                # 【端到端終點】
                t_e2e_end = time.perf_counter()

                # 紀錄量測數據 (跳過預熱期)
                if i >= self.warmups:
                    fw_latencies.append((t_fw_end - t_fw_start) * 1000)
                    e2e_latencies.append((t_e2e_end - t_e2e_start) * 1000)
                    
                    # 更新峰值記憶體
                    mem = self._get_system_memory_mb()
                    if mem > peak_memory:
                        peak_memory = mem

                if (i + 1) % 100 == 0:
                    print(f"進度: {i+1}/{total_iters}")

        finally:
            self.camera.release() 

        if fw_latencies:
            self._save_results(fw_latencies, e2e_latencies, power_mode, peak_memory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['cpu', 'cuda', 'tensorrt_fp16'], default='cuda')
    parser.add_argument('--pb', default='models/ssd_mobilenet_v3_large_coco.pb')
    parser.add_argument('--pbtxt', default='models/ssd_mobilenet_v3_large_coco.pbtxt')
    args = parser.parse_args()

    runner = BenchmarkRunner(args.pb, args.pbtxt, args.backend)
    runner.run()