#!/usr/bin/env python3
# Copyright (c) 2026 <Yanting Lin>
# Tatung University — I4210 AI實務專題

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """支援多執行緒的 HTTP 伺服器。"""
    daemon_threads = True

class MJPEGHandler(BaseHTTPRequestHandler):
    """處理 MJPEG 串流請求。"""
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with self.server.condition:
                        self.server.condition.wait()
                        frame = self.server.frame
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception:
                pass

class MJPEGServer:
    """MJPEG 串流伺服器封裝。"""
    def __init__(self, port=8080):
        self.server = ThreadedHTTPServer(('', port), MJPEGHandler)
        self.server.condition = threading.Condition()
        self.server.frame = None
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        """啟動伺服器執行緒。"""
        self.thread.start()

    def set_frame(self, frame_bytes):
        """更新目前的影像幀並通知所有客戶端。"""
        with self.server.condition:
            self.server.frame = frame_bytes
            self.server.condition.notify_all()