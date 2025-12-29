"""
YOLOv8 模型評估腳本
使用方法: python .\script\evaluate.py --model result\train\weights\best.pt --data dataset\vehicle\dataset.yaml --save-json --plots
"""

import argparse
import os
from ultralytics import YOLO
import torch

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='YOLOv8 Evaluation Script')
    parser.add_argument('--model', type=str, required=True,
                       help='訓練好的模型路徑 (.pt 檔案)')
    parser.add_argument('--data', type=str, default='dataset/vehicle/dataset.yaml',
                       help='資料集配置檔案路徑')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='評估的資料集分割')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='圖片大小')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--conf', type=float, default=0.001,
                       help='物件信心度閾值')
    parser.add_argument('--iou', type=float, default=0.6,
                       help='NMS IoU 閾值')
    parser.add_argument('--project', type=str, default='results',
                       help='結果儲存專案名稱')
    parser.add_argument('--name', type=str, default='evaluate',
                       help='實驗名稱')
    parser.add_argument('--device', type=str, default='',
                       help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-json', action='store_true',
                       help='儲存結果為 COCO JSON 格式')
    parser.add_argument('--plots', action='store_true',
                       help='產生評估圖表')
    
    return parser.parse_args()

def main():
    """主函數"""
    args = parse_args()
    
    print("📊 開始 YOLOv8 模型評估")
    print(f"🧠 模型: {args.model}")
    print(f"📊 資料集: {args.data}")
    print(f"🔍 評估分割: {args.split}")
    print(f"🖼️  圖片大小: {args.imgsz}")
    print(f"📦 批次大小: {args.batch_size}")
    
    # 檢查 GPU 可用性
    if torch.cuda.is_available():
        print(f"🎮 使用 GPU: {torch.cuda.get_device_name()}")
        if not args.device:
            args.device = 0
    else:
        print("💻 使用 CPU 評估")
        args.device = 'cpu'
    
    # 檢查模型檔案是否存在
    if not os.path.exists(args.model):
        print(f"❌ 模型檔案不存在: {args.model}")
        return
    
    # 檢查資料集配置檔案是否存在
    if not os.path.exists(args.data):
        print(f"❌ 資料集配置檔案不存在: {args.data}")
        return
    
    # 載入模型
    try:
        model = YOLO(args.model)
        print(f"✅ 成功載入模型: {args.model}")
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return
    
    # 執行評估
    try:
        results = model.val(
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch_size,
            conf=args.conf,
            iou=args.iou,
            project=args.project,
            name=args.name,
            device=args.device,
            save_json=args.save_json,
            plots=args.plots,
            verbose=True,
        )
        
        print("🎉 評估完成！")
        print(f"📁 結果儲存在: {args.project}")
        
        # 顯示評估結果
        print("\n📈 評估結果摘要:")
        print(f"   mAP50: {results.box.map50:.4f}")
        print(f"   mAP50-95: {results.box.map:.4f}")
        print(f"   Precision: {results.box.mp:.4f}")
        print(f"   Recall: {results.box.mr:.4f}")
        
        # 每個類別的結果
        if hasattr(results.box, 'ap_class_index') and len(results.box.ap_class_index) > 0:
            print("\n📊 各類別 mAP50:")
            for i, (cls_idx, ap) in enumerate(zip(results.box.ap_class_index, results.box.ap50)):
                print(f"   Class {cls_idx}: {ap:.4f}")
        
    except Exception as e:
        print(f"❌ 評估過程中發生錯誤: {e}")
        return

if __name__ == '__main__':
    main()
