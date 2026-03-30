# ==============================================================================
# Description: MJPEG streaming server (stdlib-only, OOP, thread-safe).
# Extended from course example.
# ==============================================================================

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


class _MJPEGHandler(BaseHTTPRequestHandler):
    """每個瀏覽器連線對應一個 handler instance，在各自的執行緒中執行。"""

    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.end_headers()

        server = self.server  # MJPEGServer 的 HTTPServer instance

        while True:
            # 等待新 frame（blocking），timeout 1 秒避免永久卡死
            # 若 server 關閉（_jpeg_bytes = None）則離開
            triggered = server._new_frame.wait(timeout=1.0)
            if not triggered:
                continue

            # 讀取當前 frame（tuple 整包替換，GIL atomic，不加 Lock）
            jpeg = server._jpeg_bytes
            if jpeg is None:
                break

            try:
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # 瀏覽器關閉連線，正常退出
                break

    def log_message(self, format, *args) -> None:
        """抑制每次 request 的 log 輸出。"""
        pass


class MJPEGServer:
    """
    Push-based MJPEG server。
    main thread 呼叫 push_frame() 推入 JPEG bytes，
    每個瀏覽器連線由各自的 daemon thread 處理。
    """

    DEFAULT_PORT = 8080
    JPEG_QUALITY = 80

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self._port = port

        # 共享 JPEG bytes（tuple 整包替換，GIL atomic）
        self._jpeg_bytes: bytes | None = None

        # 通知所有 handler 有新 frame
        # 用 threading.Event 廣播，不用 Queue（避免 client 數量影響 main thread）
        self._new_frame = threading.Event()

        # 建立 HTTPServer，並把共享狀態掛到 server 上讓 handler 存取
        self._httpd = HTTPServer(("0.0.0.0", port), _MJPEGHandler)
        self._httpd._new_frame   = self._new_frame
        self._httpd._jpeg_bytes  = self._jpeg_bytes

        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """在背景 daemon thread 啟動 HTTP server。"""
        if self._thread is not None and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            daemon=True,
        )
        self._thread.start()
        print(f"MJPEG server started → http://<jetson-ip>:{self._port}/")

    def push_frame(self, jpeg_bytes: bytes) -> None:
        """
        推入新的 JPEG frame，通知所有已連線的瀏覽器。
        jpeg_bytes 整包替換（GIL atomic），不加 Lock。
        set() 後不立刻 clear()，讓所有 handler 都能讀到這幀，
        下一幀 push 時 set() 會覆蓋，handler 用 timeout wait 避免錯過。
        """
        self._httpd._jpeg_bytes = jpeg_bytes
        self._new_frame.set()
        self._new_frame.clear()

    def stop(self) -> None:
        """關閉 HTTP server 並通知所有 handler 結束。"""
        # jpeg_bytes 設為 None 作為結束信號
        self._httpd._jpeg_bytes = None
        self._new_frame.set()  # 喚醒所有 wait 中的 handler

        self._httpd.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None