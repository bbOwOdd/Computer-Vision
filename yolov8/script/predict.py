"""
YOLOv8 推論/測試腳本
使用方法: python .\scripts\predict.py --model results\train\weights\best.pt --source datasets\vehicle\TrafficPolice.mp4 --save --show --save-txt --save-conf
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import torch

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='YOLOv8 Prediction Script')
    parser.add_argument('--model', type=str, required=True,
                       help='訓練好的模型路徑 (.pt 檔案)')
    parser.add_argument('--source', type=str, required=True,
                       help='輸入來源 (圖片路徑、資料夾、影片或攝影機)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='信心度閾值')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='NMS IoU 閾值')
    parser.add_argument('--project', type=str, default='results',
                       help='結果儲存專案名稱')
    parser.add_argument('--name', type=str, default='predict',
                       help='實驗名稱')
    parser.add_argument('--device', type=str, default='cpu',
                       help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save', action='store_true',
                       help='儲存預測結果')
    parser.add_argument('--show', action='store_true',
                       help='顯示預測結果')
    parser.add_argument('--save-txt', action='store_true',
                       help='儲存結果為 txt 格式')
    parser.add_argument('--save-conf', action='store_true',
                       help='在標籤中儲存信心度')   
    
    return parser.parse_args()

def main():
    """主函數"""
    args = parse_args()
    
    print("🔮 開始 YOLOv8 推論")
    print(f"🧠 模型: {args.model}")
    print(f"📁 來源: {args.source}")
    print(f"🎯 信心度閾值: {args.conf}")
    print(f"📊 IoU 閾值: {args.iou}")
    
    # 檢查 GPU 可用性
    if torch.cuda.is_available():
        print(f"🎮 使用 GPU: {torch.cuda.get_device_name()}")
        if not args.device:
            args.device = 0
    else:
        print("💻 使用 CPU 推論")
        args.device = 'cpu'
    
    # 檢查模型檔案是否存在
    if not os.path.exists(args.model):
        print(f"❌ 模型檔案不存在: {args.model}")
        return
    
    # 檢查來源是否存在
    if not os.path.exists(args.source):
        print(f"❌ 來源不存在: {args.source}")
        return
    
    # 載入模型
    try:
        model = YOLO(args.model)
        print(f"✅ 成功載入模型: {args.model}")
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return
    
    # 執行推論
    try:
        results = model.predict(
            source=args.source,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            show=args.show,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            project=args.project,
            name=args.name,
            device=args.device,
            verbose=True,
        )
        
        print("🎉 推論完成！")
        if args.save:
            print(f"📁 結果儲存在: {args.project}")
        
        # 顯示統計資訊
        total_detections = 0
        for result in results:
            if result.boxes is not None:
                total_detections += len(result.boxes)
        
        print(f"🔍 總共檢測到 {total_detections} 個物件")
        
    except Exception as e:
        print(f"❌ 推論過程中發生錯誤: {e}")
        return

if __name__ == '__main__':
    main()
