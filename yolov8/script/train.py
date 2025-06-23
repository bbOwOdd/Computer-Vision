"""
YOLOv8 訓練腳本
使用方法: python .\scripts\train.py --model models/yolov8s.pt --epochs 20 --batch-size 16 --data datasets\vehicle\dataset.yaml
"""

import argparse
import os
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='YOLOv8 Training Script')
    parser.add_argument('--data', type=str, default='datasets/vehicle/dataset.yaml', 
                       help='資料集配置檔案路徑')
    parser.add_argument('--model', type=str, default='models/yolov8s.pt', 
                       help='預訓練模型 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='訓練週期數')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='圖片大小')
    parser.add_argument('--batch-size', type=int, default=16, 
                       help='批次大小')
    parser.add_argument('--project', type=str, default='results', 
                       help='結果儲存專案名稱')
    parser.add_argument('--name', type=str, default='train', 
                       help='實驗名稱')
    parser.add_argument('--device', type=str, default='cpu', 
                       help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, 
                       help='資料載入的工作執行緒數')
    parser.add_argument('--resume', type=str, default='', 
                       help='從檢查點恢復訓練')
    parser.add_argument('--save-period', type=int, default=10, 
                       help='每 x 個 epoch 儲存檢查點')
    
    return parser.parse_args()

def check_dataset(data_path):
    """檢查資料集配置和檔案"""
    if not os.path.exists(data_path):
        print(f"❌ 資料集配置檔案不存在: {data_path}")
        return False
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 檢查必要的鍵
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data_config:
            print(f"❌ 資料集配置檔案缺少必要鍵: {key}")
            return False
    
    # 檢查路徑是否存在
    dataset_root = Path(data_config['path'])
    train_path = dataset_root / data_config['train']
    val_path = dataset_root / data_config['val']
    
    if not train_path.exists():
        print(f"❌ 訓練圖片資料夾不存在: {train_path}")
        return False
    
    if not val_path.exists():
        print(f"❌ 驗證圖片資料夾不存在: {val_path}")
        return False
    
    # 檢查是否有圖片
    train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
    val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
    
    print(f"✅ 找到 {len(train_images)} 張訓練圖片")
    print(f"✅ 找到 {len(val_images)} 張驗證圖片")
    
    if len(train_images) == 0:
        print("⚠️  警告: 訓練資料夾中沒有圖片")
    
    if len(val_images) == 0:
        print("⚠️  警告: 驗證資料夾中沒有圖片")
    
    return True

def main():
    """主函數"""
    args = parse_args()
    
    print("🚀 開始 YOLOv8 訓練")
    print(f"📊 資料集: {args.data}")
    print(f"🧠 模型: {args.model}")
    print(f"📈 訓練週期: {args.epochs}")
    print(f"🖼️  圖片大小: {args.imgsz}")
    print(f"📦 批次大小: {args.batch_size}")
    
    # 檢查 GPU 可用性
    if torch.cuda.is_available():
        print(f"🎮 使用 GPU: {torch.cuda.get_device_name()}")
        args.device = 0
    else:
        print("💻 使用 CPU 訓練")
        args.device = 'cpu'
    
    # 檢查資料集
    if not check_dataset(args.data):
        print("❌ 資料集檢查失敗，請檢查資料集配置和檔案")
        return
    
    # 載入模型
    try:
        model = YOLO(args.model)
        print(f"✅ 成功載入模型: {args.model}")
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return
    
    # 開始訓練
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            project=args.project,
            name=args.name,
            device=args.device,
            workers=args.workers,
            resume=args.resume if args.resume else False,
            save_period=args.save_period,
            verbose=True,
            plots=True,
            save=True,
            save_txt=True,
            save_conf=True,
        )
        
        print("🎉 訓練完成！")
        print(f"📁 結果儲存在: {args.project}")
        
    except Exception as e:
        print(f"❌ 訓練過程中發生錯誤: {e}")
        return

if __name__ == '__main__':
    main()
