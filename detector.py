#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>
# Tatung University — I4210 AI實務專題=============================================================================

import cv2
import numpy as np
import threading
import time
from collections import deque
from typing import Callable


class Detector:
    """Background thread object detection using SSD MobileNet V3."""

    SCALE_FACTOR:      float = 1 / 127.5
    INPUT_SIZE:        tuple = (320, 320)
    MEAN:              tuple = (127.5, 127.5, 127.5)
    NMS_IOU_THRESHOLD: float = 0.4

    COCO_LABELS = [
        "background", "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush",
    ]

    def __init__(
        self,
        model_pb: str,
        model_pbtxt: str,
        confidence_threshold: float = 0.5,
        backend: str = "cpu", 
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.labels = self.COCO_LABELS
        self.net = cv2.dnn.readNetFromTensorflow(model_pb, model_pbtxt)

        if backend == "cuda":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Backend: CUDA")
        elif backend == "tensorrt_fp16":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("Backend: TensorRT FP16")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Backend: CPU")

        self._result: tuple[np.ndarray | None, list[dict], float] | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps_history: deque = deque(maxlen=30)

    def start(self, frame_provider: Callable[[], np.ndarray | None]) -> None:
        """
        啟動背景 daemon 執行緒。
        frame_provider 由外部注入，Detector 不依賴任何 camera 實作。
        daemon=True 確保主程式結束時執行緒自動終止。
        """
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            args=(frame_provider,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """
        設定 stop_event 請求執行緒停止，等待最多 2 秒後釋放資源。
        冪等性：多次呼叫不報錯。
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self.net = None

    def get_result(self) -> tuple[np.ndarray | None, list[dict], float]:
        """
        非阻塞讀取最新推論結果。
        _result 為 tuple 整包替換（atomic），主執行緒直接讀取無需 Lock。
        若推論尚未完成第一幀，回傳符合格式的空值。
        """
        if self._result is None:
            return (None, [], 0.0)
        return self._result
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """前處理：frame → blob。"""
        return cv2.dnn.blobFromImage(
            frame,
            scalefactor=self.SCALE_FACTOR,
            size=self.INPUT_SIZE,
            mean=self.MEAN,
            swapRB=True,
        )

    def forward(self, blob: np.ndarray) -> np.ndarray:
        """純推論，不含 setInput。"""
        return self.net.forward()

    def set_input(self, blob: np.ndarray) -> None:
        """資料傳輸到推論裝置。"""
        self.net.setInput(blob)

    def postprocess(
        self, raw_dets: np.ndarray, w: int, h: int
    ) -> list[dict]:
        """後處理：raw detections → list of dict，含 NMS。"""
        boxes, confidences, class_ids = [], [], []
        for i in range(raw_dets.shape[2]):
            conf = float(raw_dets[0, 0, i, 2])
            if conf < self.confidence_threshold:
                continue
            class_id = int(raw_dets[0, 0, i, 1])
            x1 = int(raw_dets[0, 0, i, 3] * w)
            y1 = int(raw_dets[0, 0, i, 4] * h)
            x2 = int(raw_dets[0, 0, i, 5] * w)
            y2 = int(raw_dets[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(conf)
            class_ids.append(class_id)

        detections = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences,
                self.confidence_threshold,
                self.NMS_IOU_THRESHOLD,
            )
            if len(indices) > 0:
                for i in indices.flatten():
                    class_id = class_ids[i]
                    label = (
                        self.labels[class_id]
                        if class_id < len(self.labels)
                        else f"class_{class_id}"
                    )
                    detections.append({
                        "label":      label,
                        "confidence": confidences[i],
                        "bbox":       tuple(boxes[i]),
                    })
        return detections

    def _loop(self, frame_provider: Callable[[], np.ndarray | None]) -> None:
        """
        背景執行緒主體：複用 preprocess / forward / postprocess，
        避免與 benchmark 維護兩份相同邏輯。
        """
        while not self._stop_event.is_set():

            frame = frame_provider()
            if frame is None:
                time.sleep(0.005)
                continue

            t_start = time.perf_counter()

            h, w = frame.shape[:2]

            blob = self.preprocess(frame)

            self.set_input(blob)

            raw_dets = self.forward(blob)

            detections = self.postprocess(raw_dets, w, h)

            duration = time.perf_counter() - t_start
            self._fps_history.append(duration)
            avg_duration = sum(self._fps_history) / len(self._fps_history)
            current_fps  = 1.0 / avg_duration if avg_duration > 0 else 0.0

            self._result = (frame, detections, current_fps)