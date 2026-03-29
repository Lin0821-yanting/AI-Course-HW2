#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>
# Tatung University — I4210 AI實務專題

import cv2
import numpy as np
from typing import List, Tuple, Dict

class Detector:
    """SSD-MobileNet v3 物件偵測類別，支援多種運算後端。"""

    # SSD-MobileNet v3 預處理常數 [cite: 372, 385]
    INPUT_SIZE = (320, 320)
    SCALE_FACTOR = 1 / 127.5
    MEAN = (127.5, 127.5, 127.5)

    def __init__(self, pb_file: str, pbtxt_file: str, backend: str = "cuda", 
                 conf_threshold: float = 0.5, nms_threshold: float = 0.4):
        """
        初始化偵測器並載入模型。
        backend 可選: 'cpu', 'cuda', 'tensorrt_fp16' [cite: 590, 610]。
        """
        self.net = cv2.dnn.readNetFromTensorflow(pb_file, pbtxt_file)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # 設定後端與目標裝置 [cite: 591]
        if backend == "cpu":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif backend == "cuda":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        elif backend == "tensorrt_fp16":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    def preprocess(self, frame: np.ndarray):
        """將影像轉換為模型所需的 Blob 格式 [cite: 390, 511]。"""
        return cv2.dnn.blobFromImage(
            frame, 
            scalefactor=self.SCALE_FACTOR, 
            size=self.INPUT_SIZE, 
            mean=self.MEAN, 
            swapRB=True
        )

    def forward(self, blob: np.ndarray) -> np.ndarray:
        """執行模型運算（僅計時此部分以取得準確的推理延遲）。"""
        self.net.setInput(blob)
        return self.net.forward()

    def postprocessing(self, raw_dets: np.ndarray, width: int, height: int) -> List[Dict]:
        """
        處理輸出結果：解析偵測框、置信度，並套用 NMS。
        回傳包含 'bbox', 'label', 'confidence' 的清單 [cite: 388, 401, 685]。
        """
        boxes, confidences, class_ids = [], [], []
        
        # 解析輸出格式 (1, 1, 100, 7) [cite: 387, 511]
        for i in range(raw_dets.shape[2]):
            conf = float(raw_dets[0, 0, i, 2])
            if conf > self.conf_threshold:
                # 反正規化座標至像素值 [cite: 402]
                x1 = int(raw_dets[0, 0, i, 3] * width)
                y1 = int(raw_dets[0, 0, i, 4] * height)
                x2 = int(raw_dets[0, 0, i, 5] * width)
                y2 = int(raw_dets[0, 0, i, 6] * height)
                
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(conf)
                class_ids.append(int(raw_dets[0, 0, i, 1]))

        # 套用 NMS 以移除重複框 [cite: 391, 511]
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "bbox": boxes[i],      # [x, y, w, h]
                    "class_id": class_ids[i],
                    "confidence": confidences[i]
                })
        return results

    def inference(self, frame: np.ndarray) -> List[Dict]:
        """整合預處理、推理與後處理的完整流程。"""
        h, w = frame.shape[:2]
        blob = self.preprocess(frame)
        raw_dets = self.forward(blob)
        return self.postprocessing(raw_dets, w, h)