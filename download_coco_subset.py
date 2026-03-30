#!/usr/bin/env python3
# Copyright (c) 2026 <Your Name(s)>
# Tatung University 14210 AI實務專題

import os
import json
import random
import requests
import zipfile
import argparse
from typing import List, Dict, Any

class DetectorMetrics:
    """
    計算 SSD-MobileNet 偵測器準確度的類別。 [cite: 603, 625]
    """
    def __init__(self, ann_file: str, img_dir: str, model_pb: str, model_pbtxt: str):
        self.img_dir = img_dir
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        # 載入模型與初始化邏輯... [cite: 43, 50, 396]

    def run_evaluation(self):
        # 執行評估邏輯... [cite: 397]
        pass

class CocoSubsetDownloader:
    """
    用於從 COCO val2017 資料集中下載隨機圖片子集與標註的類別。 [cite: 378]
    """
    
    ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    IMG_BASE_URL = "http://images.cocodataset.org/val2017/"

    def __init__(self, output_dir: str = "coco_subset", max_images: int = 50, seed: int = None):
        """
        初始化下載器設定。 [cite: 381, 382]
        
        Args:
            output_dir: 儲存圖片與標註的目錄。
            max_images: 要下載的隨機圖片數量。
            seed: 隨機選取圖片的種子碼。
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.max_images = max_images
        self.seed = seed
        self.ann_file = os.path.join(output_dir, "instances_val2017.json")
        self.subset_ann_file = os.path.join(output_dir, "subset_annotations.json")

        # 確保目錄存在 [cite: 385]
        os.makedirs(self.images_dir, exist_ok=True)
        if seed is not None:
            random.seed(seed)

    def download_annotations(self, force: bool = False) -> None:
        """
        下載並解壓縮 COCO 標註檔。 [cite: 379, 380]
        """
        if os.path.exists(self.ann_file) and not force:
            print(f"Skipping annotations download (exists: {self.ann_file})") 
            return

        zip_path = os.path.join(self.output_dir, "annotations.zip")
        print(f"Downloading annotations from {self.ANN_URL}...")
        
        response = requests.get(self.ANN_URL, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting instances_val2017.json...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # 僅提取需要的標註檔 [cite: 380]
            zip_ref.extract("annotations/instances_val2017.json", self.output_dir)
            os.rename(os.path.join(self.output_dir, "annotations/instances_val2017.json"), self.ann_file)
            os.rmdir(os.path.join(self.output_dir, "annotations"))
        
        os.remove(zip_path)

    def select_subset(self) -> List[Dict[str, Any]]:
        """
        從原始標註中隨機選取圖片子集。 [cite: 381]
        """
        with open(self.ann_file, "r") as f:
            data = json.load(f)

        images = data["images"]
        # 隨機選取指定數量的圖片 [cite: 381]
        selected_images = random.sample(images, min(len(images), self.max_images))
        
        # 過濾標註以僅包含選定的圖片 [cite: 385]
        selected_ids = {img["id"] for img in selected_images}
        subset_data = {
            "images": selected_images,
            "annotations": [ann for ann in data["annotations"] if ann["image_id"] in selected_ids and not ann.get("iscrowd", 0)],
            "categories": data["categories"]
        }

        with open(self.subset_ann_file, "w") as f:
            json.dump(subset_data, f)
        
        return selected_images

    def download_images(self, selected_images: List[Dict[str, Any]], force: bool = False) -> None:
        """
        下載選定的圖片。 [cite: 384]
        """
        for i, img in enumerate(selected_images):
            file_name = img["file_name"]
            img_path = os.path.join(self.images_dir, file_name)
            
            if os.path.exists(img_path) and not force:
                print(f"[{i+1}/{len(selected_images)}] Skipping {file_name} (exists)") 
                continue

            print(f"[{i+1}/{len(selected_images)}] Downloading {file_name}...") 
            img_url = self.IMG_BASE_URL + file_name
            response = requests.get(img_url)
            with open(img_path, "wb") as f:
                f.write(response.content)

    def run(self, force: bool = False) -> None:
        """
        執行完整的資料準備流程。
        """
        self.download_annotations(force)
        selected = self.select_subset()
        self.download_images(selected, force)
        print(f"\nSuccessfully prepared {len(selected)} images in '{self.output_dir}'")

if __name__ == "__main__":
    import argparse
    from download_coco_subset import CocoSubsetDownloader # 引用上一步做的類別

    parser = argparse.ArgumentParser(description="SSD-MobileNet Metrics Evaluator")
    # 位置參數：自定義標註檔路徑
    parser.add_argument("ann_path", nargs="?", default=None, help="Path to COCO annotations JSON")
    # 選項參數
    parser.add_argument("--frames-dir", default=None, help="Directory containing images")
    parser.add_argument("--max-images", type=int, default=50, help="Number of images to evaluate")
    parser.add_argument("--force-download", action="store_true", help="Force re-download COCO subset")
    
    args = parser.parse_args()

    # 自動下載邏輯 [cite: 393, 481]
    target_ann = args.ann_path or "coco_subset/subset_annotations.json"
    target_img_dir = args.frames_dir or "coco_subset/images"

    if not os.path.exists(target_ann) or args.force_download:
        print("Required data not found or force-download triggered. Starting downloader...")
        downloader = CocoSubsetDownloader(max_images=args.max_images)
        downloader.run(force=args.force_download)
        target_ann = "coco_subset/subset_annotations.json"
        target_img_dir = "coco_subset/images"

    # 執行評估
    evaluator = DetectorMetrics(
        ann_file=target_ann,
        img_dir=target_img_dir,
        model_pb="models/ssd_mobilenet_v3_large_coco.pb",
        model_pbtxt="models/ssd_mobilenet_v3_large_coco.pbtxt"
    )
    evaluator.run_evaluation()