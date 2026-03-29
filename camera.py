#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name(s)>
# Tatung University — I4210 AI實務專題

import os
import cv2
import numpy as np

class Camera:
    """IMX219 CSI 相機封裝類別，支援 GStreamer pipeline。"""

    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 60):
        """
        初始化相機。
        作業二建議使用原生感測器模式 (1280x720 @ 60 FPS) [cite: 582]。
        """
        # SSH 修正：暫時移除 DISPLAY 以避免 EGL 錯誤 
        saved_display = os.environ.pop("DISPLAY", None)
        
        gst_pipeline = (
            f"nvarguscamerasrc sensor-id={camera_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"framerate={fps}/1 ! nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
        
        try:
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        finally:
            if saved_display is not None:
                os.environ["DISPLAY"] = saved_display

        if not self.cap.isOpened():
            raise RuntimeError("無法開啟 IMX219 相機，請檢查連接或重啟 nvargus-daemon。")

    def read(self) -> np.ndarray:
        """讀取即時影像幀。"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        """釋放相機資源。"""
        if self.cap.isOpened():
            self.cap.release()