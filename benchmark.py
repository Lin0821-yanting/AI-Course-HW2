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
import subprocess
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

    def __init__(
        self,
        pb_path: str,
        pbtxt_path: str,
        backend: str,
        warmups: int = 50,
        runs: int = 500,
    ):
        self.pb_path    = pb_path
        self.pbtxt_path = pbtxt_path
        self.backend    = backend
        self.warmups    = warmups
        self.runs       = runs

        self.detector = Detector(
            model_pb    = self.pb_path,
            model_pbtxt = self.pbtxt_path,
            backend     = self.backend,
        )
        self.camera = Camera(width=1280, height=720, fps=30)

    def _get_power_mode(self) -> str:
        """偵測當前 Jetson 功率模式。"""
        if jtop is None:
            return "Unknown"
        with jtop() as j:
            if j.ok:
                return j.nvpmodel.name
        return "N/A"

    def _get_system_memory_mb(self) -> float:
        """
        查詢記憶體使用量。
        先嘗試 nvidia-smi（離散 GPU），若回傳 [N/A] 則
        fallback 至 /proc/meminfo（Jetson UMA 共享記憶體）。
        
        Note: Jetson Orin Nano 使用 CPU/GPU 共享記憶體架構，
        nvidia-smi 回傳 [N/A]，因此三種 backend 的數值會相近（~5600 MB），
        反映整體系統 RAM 使用量，非 GPU 專屬 VRAM。
        """
        # 1. 先嘗試 nvidia-smi（適用離散 GPU）
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            val = result.stdout.strip()
            if val and "[N/A]" not in val:
                return float(val)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        # 2. Fallback：Jetson 共享記憶體，讀 /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                info = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        info[parts[0].rstrip(":")] = int(parts[1])
            total_kb = info.get("MemTotal", 0)
            avail_kb = info.get("MemAvailable", 0)
            return round((total_kb - avail_kb) / 1024.0, 1)
        except (FileNotFoundError, ValueError):
            # 最後 fallback：用 psutil
            import psutil
            return round(psutil.virtual_memory().used / (1024 * 1024), 1) 

    def _save_results(
        self,
        fw_latencies: list,
        e2e_latencies: list,
        power_mode: str,
        peak_mem: float,
    ) -> None:
        p50 = np.percentile(fw_latencies, 50)
        p95 = np.percentile(fw_latencies, 95)
        p99 = np.percentile(fw_latencies, 99)

        avg_e2e = np.mean(e2e_latencies)
        fps = 1000.0 / avg_e2e if avg_e2e > 0 else 0.0

        print(f"\n--- {self.backend} 結果分析 ---")
        print(f"FPS: {fps:.2f} | P50: {p50:.2f}ms | P95: {p95:.2f}ms | P99: {p99:.2f}ms")
        print(f"Memory: {peak_mem:.1f} MB (系統共享記憶體，非 GPU 專屬 VRAM)")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"benchmark_{self.backend}_{timestamp}.csv"

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Backend", "PowerMode",
                "FPS", "p50_ms", "p95_ms", "p99_ms",
                "gpu_memory_mb",  # Jetson: 整體系統 RAM，非 GPU VRAM
            ])
            writer.writerow([
                timestamp, self.backend, power_mode,
                f"{fps:.2f}", f"{p50:.2f}", f"{p95:.2f}", f"{p99:.2f}",
                f"{peak_mem:.1f}",
            ])

        print(f"結果已存至: {filename}")

    def run(self) -> None:
        """執行基準測試主迴圈。"""
        power_mode   = self._get_power_mode()
        fw_latencies  = []
        e2e_latencies = []
        peak_memory   = 0.0
        total_iters   = self.warmups + self.runs

        # 第四點優化：進度顯示粒度改為動態計算，每 10% 印一次
        log_interval = max(1, total_iters // 10)

        print(f"啟動測試: 後端={self.backend}, 模式={power_mode}")

        try:
            for i in range(total_iters):
                t_e2e_start = time.perf_counter()

                frame = self.camera.read()
                if frame is None:
                    print(f"WARNING: camera.read() 回傳 None (iter {i})，跳過此幀")
                    continue

                h, w = frame.shape[:2]

                blob = self.detector.preprocess(frame)
                self.detector.set_input(blob)   # setInput 在計時外

                t_fw_start = time.perf_counter()
                raw_dets   = self.detector.forward(blob)  # 只計 net.forward()
                t_fw_end   = time.perf_counter()

                _ = self.detector.postprocess(raw_dets, w, h)

                t_e2e_end = time.perf_counter()

                if i >= self.warmups:
                    fw_latencies.append((t_fw_end - t_fw_start) * 1000)
                    e2e_latencies.append((t_e2e_end - t_e2e_start) * 1000)

                    mem = self._get_system_memory_mb()
                    if mem > peak_memory:
                        peak_memory = mem

                # 動態進度：每 10% 印一次
                if (i + 1) % log_interval == 0:
                    print(f"進度: {i + 1}/{total_iters} ({(i + 1) / total_iters:.0%})")

        finally:
            self.camera.release()

        if fw_latencies:
            self._save_results(fw_latencies, e2e_latencies, power_mode, peak_memory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda", "tensorrt_fp16"],
        default="cuda",
    )
    parser.add_argument("--pb",    default="models/ssd_mobilenet_v3_large_coco.pb")
    parser.add_argument("--pbtxt", default="models/ssd_mobilenet_v3_large_coco.pbtxt")
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--runs",    type=int, default=500)
    args = parser.parse_args()

    runner = BenchmarkRunner(
        pb_path    = args.pb,
        pbtxt_path = args.pbtxt,
        backend    = args.backend,
        warmups    = args.warmups,
        runs       = args.runs,
    )
    runner.run()