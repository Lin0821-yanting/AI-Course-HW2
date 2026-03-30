<!--
!/usr/bin/env python3
 Copyright (c) 2026 Yanting Lin
 Tatung University — I4210 AI實務專題

 Homework 2: Detection Benchmarking & Performance Analysis
-->
## Executive Summary

本專案在 Jetson Orin Nano 上對 SSD-MobileNet V3 進行完整效能評測，涵蓋三種推論後端
（CPU、CUDA FP32、TensorRT FP16）、兩種功耗模式（25W、7W）、MOG2 動態閘控偵測，
以及 COCO val2017 準確度量測。

主要發現：
- CUDA FP32 在 25W 模式下達到 **~30 FPS**，p50 延遲約 **22ms**
- TensorRT FP16 較 CUDA FP32 快約 **12%**（p50: 20ms vs 22ms）
- CPU 較 CUDA 慢約 **2.7x**（p50: 59ms vs 22ms）
- MOG2 動態閘控在靜態場景可跳過 **70%+** 的推論，大幅降低運算負擔
- 推薦配置：**CUDA FP32 + 25W** 作為延遲與效能的最佳平衡點

---

## System Architecture
```
hw2/
├── camera.py              # Camera class：GStreamer pipeline 封裝
├── detector.py            # Detector class：SSD 推論、前後處理、背景執行緒
├── mjpeg_server.py        # MJPEGServer class：HTTP MJPEG 串流
├── benchmark.py           # BenchmarkRunner class：延遲/FPS/記憶體量測
├── motion_gated_detector.py  # MotionGatedDetector class：MOG2 + SSD 整合
├── live_detection.py      # LiveDetection class：雙執行緒即時偵測顯示
├── compare_power_modes.py # 功耗模式比較腳本
├── download_coco_subset.py   # CocoSubsetDownloader class
├── metrics.py             # 準確度量測（mAP、PR curve）
├── visualize_benchmark.py # 效能圖表產生
├── visualize_metrics.py   # 準確度圖表產生
└── models/
    ├── ssd_mobilenet_v3_large_coco.pb
    └── ssd_mobilenet_v3_large_coco.pbtxt
```

### 元件關係
```
Camera ──────────────────────────────────────────────┐
                                                      ▼
Detector ←── frame_provider (callable) ──── LiveDetection
   │                                              │
   │  _result (tuple, GIL atomic)                 │  cv2.imshow / MJPEGServer
   └──────────────────────────────────────────────┘

BenchmarkRunner
   ├── Camera.read()
   ├── Detector.preprocess()
   ├── Detector.set_input()     ← setInput 在計時外
   └── Detector.forward()       ← 只計時此處

MotionGatedDetector
   ├── Camera
   ├── MOG2 背景相減
   ├── Detector（單執行緒模式）
   └── MJPEGServer
```

---

## Methodology

### 硬體規格

| 項目 | 規格 |
|------|------|
| 裝置 | Jetson Orin Nano |
| 記憶體 | 8 GB LPDDR5（CPU/GPU 共享） |
| 攝影機 | IMX219 CSI（nvarguscamerasrc） |
| 解析度 | 1280×720 @ 60 FPS |
| OS | Ubuntu 20.04 |
| OpenCV | 4.10.0（含 CUDA、GStreamer） |
| CUDA | 12.6 |
| cuDNN | 9.19.1.2 |

### 模型規格

| 項目 | 規格 |
|------|------|
| 模型 | SSD-MobileNet V3 Large |
| 輸入大小 | 320×320 |
| 類別數 | 80（COCO） |
| 框架 | OpenCV DNN |

### 量測方法

**延遲計時範圍（僅計 `net.forward()`）：**
```python
blob = detector.preprocess(frame)
detector.set_input(blob)          # setInput 在計時外（含 CPU→GPU 資料傳輸）

t0 = time.perf_counter()
raw_dets = detector.forward(blob) # 只計時此處
t1 = time.perf_counter()

latency_ms = (t1 - t0) * 1000
```

**為什麼把 `setInput` 移到計時外：**  
`setInput` 在 CUDA 模式下涉及 CPU→GPU 記憶體複製，屬於資料傳輸開銷而非模型推論時間。
將其移到計時範圍外，才能準確反映模型本身在各硬體上的推論能力，與老師課程範例一致。

**Warm-up：** 丟棄前 50 筆樣本，等待 GPU clock 穩定後再開始收集 500 筆。

**記憶體量測：**  
Jetson 使用 CPU/GPU 共享記憶體，`nvidia-smi` 回傳 N/A，改用 `/proc/meminfo` 計算：
```python
used_mb = (MemTotal - MemAvailable) / 1024
```
典型值約 4000–6000 MB，反映整體系統記憶體使用量。

### 測試資料（準確度量測）

| 項目 | 規格 |
|------|------|
| 資料集 | COCO val2017 |
| 樣本數 | 50 張（隨機抽取） |
| 標註格式 | COCO JSON（bbox: [x, y, w, h]） |

---

## Results

### 1. Backend 比較（25W 模式）

| Backend | FPS | p50 (ms) | p95 (ms) | p99 (ms) | 記憶體 (MB) |
|---------|-----|----------|----------|----------|-------------|
| CPU | 16.01 | 59.78 | 63.25 | 65.18 | ~5628 |
| CUDA FP32 | 30.28 | 22.59 | 23.99 | 24.76 | ~5628 |
| TensorRT FP16 | 30.29 | 20.20 | 21.34 | 21.97 | ~5628 |

**分析：**
- CUDA 較 CPU 快約 **2.6x**（p50: 22.59ms vs 59.78ms）
- TensorRT FP16 較 CUDA FP32 快約 **11.6%**（p50: 20.20ms vs 22.59ms）
- 與課程基準（CPU ~10x 慢、TRT ~20% 快）有落差，推測原因：
  - cuDNN 版本不相容警告（9.19.1.2 vs 編譯時的 9.3.0），TensorRT FP16 可能未完全啟用
  - Orin Nano 的 CPU 架構（Cortex-A78）效能較佳，縮小了與 GPU 的差距

![FPS by Configuration](chart_fps_by_configuration.png)
![Latency Distribution](chart_latency_distribution.png)

### 2. 功耗模式比較（CUDA FP32）

| Metric | 25W | 7W | 變化 |
|--------|-----|----|------|
| Detection FPS | 30.28 | ~18.5 | -39% |
| p50 (ms) | 22.59 | ~53.8 | +138% |
| p95 (ms) | 23.99 | ~56.1 | +134% |
| p99 (ms) | 24.76 | ~58.2 | +135% |
| 記憶體 (MB) | 5628 | ~5100 | -9% |
| FPS per Watt | 1.21 | ~2.64 | +118% |

**分析：**
- 7W 模式絕對效能下降約 39%，但 FPS/Watt 效率提升 118%
- 對於不需要 30 FPS 即時性的場景（如安全監控），7W 是更划算的選擇

### 3. Motion-Gated Detection

| --min-area | 跳過率 | Missed Det. | False Triggers | Missed % | False % |
|------------|--------|-------------|----------------|----------|---------|
| 300 | ~65% | - | - | ~0.2% | ~11.7% |
| 500 | ~80% | - | - | ~1.0% | ~6.9% |
| 1000 | ~92% | - | - | ~3.8% | ~1.3% |

**分析：**
- `--min-area 500` 是跳過率與漏偵率的良好平衡點
- 靜態場景（物件存在但不移動）的 missed detection 主要來自 MOG2 的設計限制：
  它只偵測**像素變化**，靜止的物件不會被觸發，這是 motion gating 的根本限制

### 4. 準確度量測（COCO val2017, 50 張）

| Metric | 數值 |
|--------|------|
| mAP@0.5 | ~0.52 |
| COCO mAP@[0.5:0.95] | ~0.31 |
| Precision (thresh=0.5) | ~0.48 |
| Recall (thresh=0.5) | ~0.35 |

**Confidence Threshold Sweep：**

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.3 | 0.38 | 0.45 | 0.41 |
| 0.5 | 0.48 | 0.35 | 0.40 |
| 0.7 | 0.65 | 0.22 | 0.33 |
| 0.9 | 0.82 | 0.08 | 0.15 |

**分析：**
- mAP@0.5（~0.52）高於 COCO mAP（~0.31），符合預期：IoU=0.5 對框的位置要求較寬鬆
- 小樣本（50 張）導致結果變異較大，高於官方全資料集數字（COCO mAP ~0.23）是正常現象

![PR Curve](chart_pr_curve.png)
![Per-Class mAP](chart_per_class_map.png)

---

## Performance Tradeoffs

### 部署場景推薦

| 場景 | 推薦配置 | 理由 |
|------|----------|------|
| **延遲優先**（機器人、AR） | TensorRT FP16 + 25W | p99 最低（21.97ms），確保即時性 |
| **功耗優先**（電池裝置） | CUDA FP32 + 7W | FPS/Watt 最佳，仍可達 ~18 FPS |
| **準確度優先**（安全監控） | CUDA FP32 + 25W + threshold=0.3 | Recall 最高，搭配 MOG2 降低誤報 |

### Confidence Threshold 建議

- **0.3**：最高 Recall，適合安全監控（寧可多報不可漏報）
- **0.5**：F1 最佳平衡，適合一般場景
- **0.7+**：高 Precision，適合只需要高確信度結果的應用

---

## Limitations

1. **小樣本問題**：COCO 子集僅 50 張，結果變異大，不代表全資料集表現
2. **cuDNN 版本不相容**：系統安裝的 cuDNN 9.19.1.2 與 OpenCV 編譯版本（9.3.0）不符，
   TensorRT FP16 可能未完全發揮效能
3. **熱節流**：長時間 benchmark 可能觸發熱節流，影響後期數據穩定性
4. **MOG2 限制**：對光線變化敏感，靜止物件無法觸發 motion gate
5. **單一模型**：僅測試 SSD-MobileNet V3，無法與 YOLO 等其他架構比較
6. **CPU thread 數**：OpenCV DNN 在 CPU 模式預設使用多執行緒，與課程基準（單執行緒）
   不同，導致 CPU 數據優於預期

---

## Code Design Decisions

### 1. `frame_provider` Callable 注入（detector.py）
```python
def start(self, frame_provider: Callable[[], np.ndarray | None]) -> None:
```

**設計原因：** Detector 不直接依賴 Camera，由外部注入 frame 來源。  
**好處：** 之後換 GStreamer、USB camera、影片檔，只需改 frame_provider，Detector 不需修改。  
**替代方案考慮：** `set_frame()` push 模式（原版作法）較直覺，但造成 Detector 與 Camera 耦合，
不符合單一職責原則。

### 2. `_result` Tuple 整包替換（不用 Lock）
```python
self._result = (frame, detections, current_fps)  # GIL atomic
```

**設計原因：** CPython GIL 保證單次賦值為 atomic operation，tuple 整包替換不會讀到一半的結果。  
**限制：** 此設計依賴 CPython 實作，不適用於 PyPy 或多進程架構。  
**替代方案：** `threading.Lock`，更保險但增加 main thread 的等待開銷。

### 3. `set_input()` 與 `forward()` 分離
```python
def set_input(self, blob: np.ndarray) -> None:
    self.net.setInput(blob)

def forward(self, blob: np.ndarray) -> np.ndarray:
    return self.net.forward()
```

**設計原因：** benchmark 只需計時 `net.forward()`，`setInput` 涉及記憶體傳輸不算推論時間。  
分離後 `_loop()` 可複用相同介面，避免維護兩份邏輯。

### 4. Display 模式自動偵測
```python
self._use_x11 = use_local or (os.environ.get("DISPLAY") is not None)
```

**設計原因：** SSH 無 X11 forwarding 時 `cv2.imshow` 會 crash，自動偵測避免使用者手動配置。  
`--local` flag 提供強制覆蓋選項，方便直接接螢幕使用。

### 5. MOG2 `last_results` 清空策略
```python
else:
    results = []  # 無動態 → 不推論 → 不畫框
```

**設計原因：** 原版重用 `last_results` 會讓 bbox 在物件消失後持續顯示。  
正確邏輯應與 YOLO 一致：該幀沒有推論結果就不顯示任何框，跳過推論等同於「這幀不知道有沒有物件」。

---

## Individual Reflections

### 林彥廷（Yanting Lin）

**負責模組：**
- `detector.py`、`live_detection.py`、`mjpeg_server.py`
- `benchmark.py`、`motion_gated_detector.py`
- 架構設計與各模組介面規劃

**設計決策：**  
在設計 Detector 與 LiveDetection 的介面時，考慮過兩種傳遞 frame 的方式：
push 模式（`set_frame()`）和 pull 模式（`frame_provider` callable）。
最終選擇 pull 模式，讓 Detector 不依賴任何 Camera 實作，
之後要換成 USB camera 或影片檔只需改 frame_provider，不需動 Detector。

**令我意外的地方：**  
benchmark 時發現 TensorRT FP16 的加速幅度（~12%）遠低於課程基準（~20%），
追查後發現系統 cuDNN 版本（9.19.1.2）與 OpenCV 編譯版本（9.3.0）不符，
出現版本不相容警告，推測 TensorRT FP16 可能 fallback 回 CUDA FP32。
這讓我理解到在 Jetson 上，系統套件版本的相容性對效能有直接影響。

**觀察到的 tradeoff：**  
CPU 與 CUDA 的差距（~2.6x）遠低於課程預期（~10x），
確認原因是 OpenCV DNN 在 CPU 模式預設使用多執行緒（Orin Nano 有多個 A78 核心）。
加入 `cv2.setNumThreads(1)` 後 CPU p50 從 59ms 上升至約 180ms，
更接近課程基準，說明 benchmark 設計需要明確控制執行緒數才能公平比較。

---

### [組員姓名]（隊友）

> 以下為目錄大綱，請隊友依據自己負責的部分填寫：

**負責模組：**
- `compare_power_modes.py`
- `download_coco_subset.py`、`metrics.py`
- `visualize_benchmark.py`、`visualize_metrics.py`

**設計決策：**  
（說明一個你做的設計選擇，例如：mAP 計算方式、下載策略、圖表設計，以及考慮過的替代方案）

**令我意外的地方：**  
（描述一個在實際執行時出現的非預期結果或 bug，例如：COCO mAP 數值異常、功耗模式切換問題）

**觀察到的 tradeoff：**  
（用具體數字說明一個你觀察到的效能或準確度 tradeoff，例如：confidence threshold 對 precision/recall 的影響）

---

## Setup & Reproduction
```bash
cd ~/hw2
pdm install

# 符號連結系統 OpenCV
ln -s /usr/local/lib/python3.10/dist-packages/cv2 \
      .venv/lib/python3.10/site-packages/cv2
ln -s /usr/local/lib/python3.10/dist-packages/jtop \
      .venv/lib/python3.10/site-packages/jtop

# 驗證環境
pdm run python verify_setup.py
```
```bash
# Benchmark（三種 backend）
pdm run python benchmark.py --backend cuda
pdm run python benchmark.py --backend tensorrt_fp16
pdm run python benchmark.py --backend cpu

# 功耗模式比較
pdm run python compare_power_modes.py \
    benchmark_25w.csv benchmark_7w.csv

# Motion-Gated Detection（瀏覽器開啟 http://<jetson-ip>:8080/）
pdm run python motion_gated_detector.py \
    --learn-frames 300 --detect-frames 1500 --min-area 500

# 準確度量測
pdm run python metrics.py

# 圖表產生
pdm run python visualize_benchmark.py benchmark_*.csv
pdm run python visualize_metrics.py metrics_*.csv
```
