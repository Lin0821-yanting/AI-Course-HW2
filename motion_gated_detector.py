#!/usr/bin/env python3
# Copyright (c) 2026 Yanting Lin
# Tatung University — I4210 AI實務專題

import cv2
import numpy as np
import time
import argparse
from camera import Camera
from detector import Detector
from mjpeg_server import MJPEGServer

WARMUP_FRAMES = 50  # 丟棄前 50 筆推論，等待 GPU 穩定


class MotionGatedDetector:
    """動態閘控偵測系統。"""

    def __init__(self, args):
        self.args = args
        self.camera = Camera(width=1280, height=720, fps=30)

        self.detector = Detector(
            model_pb             = args.pb,
            model_pbtxt          = args.pbtxt,
            confidence_threshold = 0.3,
            backend              = args.backend,
        )
        self.server = MJPEGServer(port=8080)

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=args.learn_frames, varThreshold=50, detectShadows=True
        )
        self.erode_kernel  = np.ones((3, 3), np.uint8)
        self.dilate_kernel = np.ones((7, 7), np.uint8)

        # 統計數據
        self.inference_latencies = []
        self.inference_count     = 0
        self.missed_detections   = 0
        self.false_triggers      = 0

        # motion trigger / static frame 計數（對齊老師輸出格式）
        self.motion_trigger_count = 0
        self.static_frame_count   = 0

        # warmup 計數：前 WARMUP_FRAMES 筆推論不計入 latency
        self._warmup_done = 0

    def _cleanup_mask(self, mask: np.ndarray) -> np.ndarray:
        """形態學清理：侵蝕 3x3 移除雜訊，膨脹 7x7 填補空隙。"""
        mask = cv2.erode(mask, self.erode_kernel, iterations=1)
        mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)
        return mask

    def _run_inference(self, frame: np.ndarray) -> tuple[list[dict], float]:
        """
        執行推論並回傳 (detections, latency_ms)。
        warmup 期間推論正常執行但不記錄 latency，回傳 -1.0 作為標記。
        """
        h, w = frame.shape[:2]
        blob = self.detector.preprocess(frame)
        self.detector.set_input(blob)          # setInput 在計時外

        t0       = time.perf_counter()
        raw_dets = self.detector.forward(blob) # 只計時 net.forward()
        latency  = (time.perf_counter() - t0) * 1000

        results = self.detector.postprocess(raw_dets, w, h)

        if self._warmup_done < WARMUP_FRAMES:
            # warmup 期間不記錄 latency，讓 GPU 先穩定
            self._warmup_done += 1
            return results, -1.0

        return results, latency

    def run(self) -> None:
        """執行偵測流程。"""
        self.server.start()
        print(f"Backend: {self.args.backend.upper()}")
        print(f"Warming up GPU ({WARMUP_FRAMES} frames)...")
        print(f"Learning background ({self.args.learn_frames} frames)... keep still.")

        total_frames = self.args.learn_frames + self.args.detect_frames

        for i in range(total_frames):
            frame = self.camera.read()
            if frame is None:
                break

            is_learning = i < self.args.learn_frames
            mask = self.bg_sub.apply(frame)

            if is_learning:
                # learning 期間同步做 GPU warmup（不計入 latency）
                if self._warmup_done < WARMUP_FRAMES:
                    self._run_inference(frame)  # latency 回傳 -1.0，自動忽略

                cv2.putText(
                    frame, "LEARNING BACKGROUND...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )
                self._stream_frame(frame)
                continue

            # --- Detection Phase ---
            cleaned_mask = self._cleanup_mask(mask)
            contours, _  = cv2.findContours(
                cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            motion_gate = any(cv2.contourArea(c) > self.args.min_area for c in contours)
            run_ssd     = motion_gate or self.args.evaluate
            results     = []

            # 統計 motion trigger / static frame
            if motion_gate:
                self.motion_trigger_count += 1
            else:
                self.static_frame_count += 1

            if run_ssd:
                results, latency = self._run_inference(frame)
                self.inference_count += 1

                # warmup 結束後才記錄 latency
                if latency >= 0:
                    self.inference_latencies.append(latency)
            else:
                results = []

            # Evaluation Mode Metrics
            if self.args.evaluate:
                object_present = any(d["confidence"] > 0.5 for d in results)
                if not motion_gate and object_present:
                    self.missed_detections += 1
                if motion_gate and not object_present:
                    self.false_triggers += 1

            self._visualize_and_stream(frame, results, contours, motion_gate)

        self._print_summary(total_frames)
        self._save_csv(total_frames)
        self.camera.release()

    def _visualize_and_stream(
        self,
        frame: np.ndarray,
        results: list[dict],
        contours: list,
        gate: bool,
    ) -> None:
        """繪製視覺化結果並推播至串流。"""
        for c in contours:
            if cv2.contourArea(c) > self.args.min_area:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        for det in results:
            x, y, w, h = det["bbox"]
            label      = det["label"]
            confidence = det["confidence"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label} {confidence:.0%}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        status = "RUN"  if gate else "SKIP"
        color  = (0, 255, 0) if gate else (0, 0, 255)
        cv2.putText(
            frame, f"GATE: {status}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
        self._stream_frame(frame)

    def _stream_frame(self, frame: np.ndarray) -> None:
        """JPEG 編碼並推播至 MJPEG server。"""
        _, buf = cv2.imencode(".jpg", frame)
        self.server.push_frame(buf.tobytes())

    def _print_summary(self, total: int) -> None:
        """輸出最終統計報告，格式對齊老師範例。"""
        detect_frames = self.args.detect_frames 

        if self.args.evaluate:
            skip_rate = (
                self.static_frame_count / detect_frames * 100
                if detect_frames > 0 else 0.0
            )
            skip_label = "Skip rate (gate would skip):"
        else:
            skip_rate = (
                (1 - self.inference_count / detect_frames) * 100
                if detect_frames > 0 else 0.0
            )
        skip_label = "Skip rate:                :"
        p50 = np.percentile(self.inference_latencies, 50) if self.inference_latencies else 0
        p95 = np.percentile(self.inference_latencies, 95) if self.inference_latencies else 0
        p99 = np.percentile(self.inference_latencies, 99) if self.inference_latencies else 0

        if self.args.evaluate:
            missed_str = (
                f"{self.missed_detections} "
                f"(skipped but objects present)"
            )
            false_str = (
                f"{self.false_triggers} "
                f"(triggered but no objects)"
            )
            missed_pct = f"{self.missed_detections / detect_frames * 100:.1f}%"
            false_pct  = f"{self.false_triggers  / detect_frames * 100:.1f}%"
        else:
            missed_str = missed_pct = "N/A (需要 --evaluate 模式)"
            false_str  = false_pct  = "N/A (需要 --evaluate 模式)"

        # 對齊老師的輸出格式
        title = "Motion-Gated Evaluation Summary" if self.args.evaluate \
                else "Motion-Gated Detection Summary"

        print(f"\n--- {title} ---")
        print(f"Learn frames:      {self.args.learn_frames}")
        print(f"Detection frames:  {detect_frames} / {detect_frames}")
        print(f"Motion triggers:   {self.motion_trigger_count} (gate said \"run\")")
        print(f"Static frames:     {self.static_frame_count} (gate said \"skip\")")
        print(f"Missed detections: {missed_str}")
        print(f"False triggers:    {false_str}")
        print(f"{skip_label}  {skip_rate:.1f}%")  # ← 修正
        print(f"Missed detection %:{missed_pct}")
        print(f"False trigger %:   {false_pct}")
        print(f"Inference p50:     {p50:.1f} ms")
        print(f"Inference p95:     {p95:.1f} ms")
        print(f"Inference p99:     {p99:.1f} ms")

    def _save_csv(self, total: int) -> None:
        """將量測結果存至 CSV。"""
        from datetime import datetime
        import csv

        detect_frames = self.args.detect_frames
        skip_rate = (
            (1 - self.inference_count / detect_frames) * 100
            if detect_frames > 0 else 0.0
        )
        p50 = np.percentile(self.inference_latencies, 50) if self.inference_latencies else 0
        p95 = np.percentile(self.inference_latencies, 95) if self.inference_latencies else 0
        p99 = np.percentile(self.inference_latencies, 99) if self.inference_latencies else 0

        if self.args.evaluate:
            missed_val     = self.missed_detections
            false_val      = self.false_triggers
            missed_pct_val = f"{self.missed_detections / detect_frames * 100:.1f}"
            false_pct_val  = f"{self.false_triggers  / detect_frames * 100:.1f}"
        else:
            missed_val = false_val = "N/A"
            missed_pct_val = false_pct_val = "N/A"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"motion_gated_{self.args.backend}_{timestamp}.csv"

        with open(filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Backend", "MinArea", "EvaluateMode",
                "LearnFrames", "DetectFrames", "InferenceCalls",
                "MotionTriggers", "StaticFrames",
                "SkipRate_%", "p50_ms", "p95_ms", "p99_ms",
                "MissedDetections", "MissedPct_%",
                "FalseTriggers", "FalsePct_%",
            ])
            writer.writerow([
                timestamp, self.args.backend, self.args.min_area,
                self.args.evaluate,
                self.args.learn_frames, detect_frames, self.inference_count,
                self.motion_trigger_count, self.static_frame_count,
                f"{skip_rate:.1f}", f"{p50:.2f}", f"{p95:.2f}", f"{p99:.2f}",
                missed_val, missed_pct_val,
                false_val, false_pct_val,
            ])

        print(f"結果已存至: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda", "tensorrt_fp16"],
        default="cuda",
    )
    parser.add_argument("--pb",           default="models/ssd_mobilenet_v3_large_coco.pb")
    parser.add_argument("--pbtxt",        default="models/ssd_mobilenet_v3_large_coco.pbtxt")
    parser.add_argument("--learn-frames", type=int, default=300)
    parser.add_argument("--detect-frames",type=int, default=1500)
    parser.add_argument("--min-area",     type=int, default=500)
    parser.add_argument("--evaluate",     action="store_true")
    args = parser.parse_args()

    system = MotionGatedDetector(args)
    system.run()