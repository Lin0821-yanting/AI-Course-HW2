# ==============================================================================
# Description: Live detection with dual display mode (X11 / MJPEG server).
# ==============================================================================

import argparse
import os
import time
import cv2
import numpy as np
from collections import deque
from detector import Detector
from mjpeg_server import MJPEGServer


class LiveDetection:
    """
    Main thread: capture frames at 30 FPS, display with latest detections.
    Display mode:
        - DISPLAY 環境變數存在 → cv2.imshow (X11)
        - 無 DISPLAY            → MJPEG server（瀏覽器觀看）
    """

    # Camera defaults
    DEFAULT_SENSOR_ID = 0
    DEFAULT_WIDTH     = 1280
    DEFAULT_HEIGHT    = 720
    CAMERA_FPS        = 30

    # Drawing constants
    LABEL_COLORS = {
        "person":  (0, 255, 0),
        "car":     (255, 0, 0),
        "bicycle": (0, 165, 255),
        "dog":     (0, 0, 255),
    }
    DEFAULT_COLOR    = (200, 200, 200)
    BOX_THICKNESS    = 2
    LABEL_FONT_SCALE = 0.5
    HUD_FONT_SCALE   = 0.7
    HUD_ORIGIN       = (10, 30)
    HUD_LINE_SPACING = 28
    JPEG_QUALITY     = 80

    def __init__(
        self,
        model_pb: str,
        model_pbtxt: str,
        sensor_id: int              = DEFAULT_SENSOR_ID,
        width: int                  = DEFAULT_WIDTH,
        height: int                 = DEFAULT_HEIGHT,
        confidence_threshold: float = 0.5,
        port: int                   = MJPEGServer.DEFAULT_PORT,
        use_local: bool             = False,  # ← 新增
    ) -> None:
        self.detector = Detector(model_pb, model_pbtxt, confidence_threshold)
        self._latest_frame: np.ndarray | None = None

        # --local 優先，否則自動偵測 DISPLAY
        self._use_x11 = use_local or (os.environ.get("DISPLAY") is not None)

        if self._use_x11:
            print("Display mode: X11 (cv2.imshow)")
            self._server = None
        else:
            print("Display mode: MJPEG server")
            self._server = MJPEGServer(port=port)

        self.cap = self._open_camera(sensor_id, width, height)
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {self.width}x{self.height}")

    def _open_camera(
        self, sensor_id: int, width: int, height: int, max_retry: int = 3
    ) -> cv2.VideoCapture:
        """建立 GStreamer pipeline，失敗最多重試 max_retry 次。"""
        pipeline = (
            f"nvarguscamerasrc sensor-id={sensor_id} "
            f"! video/x-raw(memory:NVMM),"
            f"width={width},height={height},"
            f"framerate={self.CAMERA_FPS}/1 "
            f"! nvvidconv ! video/x-raw,format=BGRx "
            f"! videoconvert ! video/x-raw,format=BGR "
            f"! appsink drop=1"
        )
        for attempt in range(1, max_retry + 1):
            # SSH workaround：暫時移除 DISPLAY 避免 EGL 衝突
            saved = os.environ.pop("DISPLAY", None)
            try:
                cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            finally:
                if saved is not None:
                    os.environ["DISPLAY"] = saved

            if cap.isOpened():
                return cap

            print(f"Camera open failed (attempt {attempt}/{max_retry}), retrying...")
            time.sleep(2.0)

        raise RuntimeError(
            "Cannot open camera after retries.\n"
            "Check: ls /dev/video*\n"
            "Try:   sudo systemctl restart nvargus-daemon"
        )

    def _frame_provider(self) -> np.ndarray | None:
        """Detector thread 呼叫此 callable 取得最新 frame（GIL atomic 讀取）。"""
        return self._latest_frame

    def _draw(self, frame: np.ndarray, detections: list[dict],
              display_fps: float, detect_fps: float) -> None:
        """在 frame 上畫 bounding box 與 HUD。"""
        for det in detections:
            label       = det["label"]
            confidence  = det["confidence"]
            x, y, w, h  = det["bbox"]
            color = self.LABEL_COLORS.get(label, self.DEFAULT_COLOR)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, self.BOX_THICKNESS)

            text = f"{label} {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX,
                self.LABEL_FONT_SCALE, self.BOX_THICKNESS,
            )
            cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(
                frame, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, self.LABEL_FONT_SCALE,
                (0, 0, 0), 1,
            )

        x0, y0 = self.HUD_ORIGIN
        cv2.putText(
            frame, f"Display FPS: {display_fps:.1f}",
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX, self.HUD_FONT_SCALE,
            (0, 255, 255), 1,
        )
        cv2.putText(
            frame, f"Detect FPS:  {detect_fps:.1f}",
            (x0, y0 + self.HUD_LINE_SPACING),
            cv2.FONT_HERSHEY_SIMPLEX, self.HUD_FONT_SCALE,
            (0, 255, 255), 1,
        )
        cv2.putText(
            frame, f"Objects: {len(detections)}",
            (x0, y0 + self.HUD_LINE_SPACING * 2),
            cv2.FONT_HERSHEY_SIMPLEX, self.HUD_FONT_SCALE,
            (0, 255, 255), 1,
        )

    def run(self) -> None:
        """啟動 Detector thread，main thread 以 cap.read() 節拍跑顯示迴圈。"""
        self.detector.start(frame_provider=self._frame_provider)

        if self._server is not None:
            self._server.start()

        display_fps_history: deque = deque(maxlen=30)
        last_frame_time = time.perf_counter()

        mode = "X11 — press 'q' to quit" if self._use_x11 else "MJPEG — press Ctrl+C to stop"
        print(f"Live Detection ({mode})")

        try:
            while True:
                # cap.read() 阻塞等 camera，本身就是 30 FPS 節拍器
                ret, frame = self.cap.read()
                if not ret:
                    print("ERROR: Cannot read frame")
                    break

                # --- Display FPS（幀與幀實際間隔，含 cap.read() 等待）---
                now = time.perf_counter()
                real_elapsed = now - last_frame_time
                last_frame_time = now
                if real_elapsed > 0:
                    display_fps_history.append(real_elapsed)
                avg = sum(display_fps_history) / len(display_fps_history)
                display_fps = 1.0 / avg if avg > 0 else 0.0

                # --- 更新共享 frame（GIL atomic）---
                self._latest_frame = frame

                # --- 取推論結果（non-blocking）---
                _, detections, detect_fps = self.detector.get_result()

                # --- 畫框與 HUD ---
                self._draw(frame, detections, display_fps, detect_fps)

                # --- 顯示 ---
                if self._use_x11:
                    cv2.imshow("Live Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    _, buf = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY],
                    )
                    self._server.push_frame(buf.tobytes())

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """依序釋放所有資源。"""
        self.detector.stop()
        self.cap.release()
        if self._server is not None:
            self._server.stop()
        if self._use_x11:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live object detection")
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Use cv2.imshow (X11) instead of MJPEG server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=MJPEGServer.DEFAULT_PORT,
        help=f"MJPEG server port (default: {MJPEGServer.DEFAULT_PORT})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)",
    )
    args = parser.parse_args()

    live = LiveDetection(
        model_pb             = "models/ssd_mobilenet_v3_large_coco.pb",
        model_pbtxt          = "models/ssd_mobilenet_v3_large_coco.pbtxt",
        use_local            = args.local,
        port                 = args.port,
        confidence_threshold = args.confidence,
    )
    try:
        live.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")