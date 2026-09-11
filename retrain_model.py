"""
PPE Detection Model Retraining Script
======================================
This script retrains the YOLOv8 model with OPTIMAL settings for much better accuracy.

The original model was trained with:
  - pretrained=False (no transfer learning!)
  - yolov8n (nano - smallest, weakest model)
  - mosaic=0.0 (no mosaic augmentation)
  - mixup=0.0 (no mixup augmentation)

This script fixes ALL of those issues:
  - pretrained=True (COCO transfer learning - HUGE boost)
  - yolov8s (small - 2x more accurate than nano)
  - Full augmentations enabled
  - Proper hyperparameters

Usage:
------
Step 1: Get your Roboflow API key from https://app.roboflow.com/settings/api
Step 2: Run: python retrain_model.py --api-key YOUR_API_KEY
Step 3: Wait for training to complete (~30-60 minutes on GPU, longer on CPU)
Step 4: The new model will be saved to models/best.pt automatically
"""

import argparse
import os
import shutil
from pathlib import Path


def download_dataset(api_key):
    """Download the Construction Site Safety dataset from Roboflow"""
    print("=" * 60)
    print("📥 Downloading Construction Site Safety Dataset...")
    print("=" * 60)

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
    version = project.version(28)
    dataset = version.download("yolov8", location="./dataset")

    print(f"✅ Dataset downloaded to: ./dataset")
    return "./dataset/data.yaml"


def train_model(data_yaml, epochs=50, model_size="s", batch=16, imgsz=640):
    """Train YOLOv8 with optimal settings"""
    from ultralytics import YOLO

    # Choose model size
    model_name = f"yolov8{model_size}.pt"
    print(f"\n{'=' * 60}")
    print(f"🚀 Training YOLOv8-{model_size.upper()} with optimal settings")
    print(f"{'=' * 60}")
    print(f"  Model:       {model_name} (pretrained on COCO)")
    print(f"  Dataset:     {data_yaml}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch}")
    print(f"  Image size:  {imgsz}")
    print(f"  Augments:    mosaic, mixup, fliplr, hsv, scale, translate")
    print(f"{'=' * 60}\n")

    # Load pretrained model (transfer learning from COCO)
    model = YOLO(model_name)

    # Train with optimal settings
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        pretrained=True,        # CRITICAL: use COCO pretrained weights
        optimizer="AdamW",      # Better optimizer than SGD for fine-tuning
        lr0=0.001,              # Lower initial LR for fine-tuning
        lrf=0.01,               # Final LR ratio
        warmup_epochs=5,        # Warm up learning rate
        patience=20,            # More patience before early stopping
        cos_lr=True,            # Cosine LR schedule (smoother)

        # Augmentations (CRITICAL for generalization)
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.15,             # Mixup augmentation
        fliplr=0.5,             # Horizontal flip
        hsv_h=0.015,            # Hue augmentation
        hsv_s=0.7,              # Saturation augmentation
        hsv_v=0.4,              # Value/brightness augmentation
        degrees=10.0,           # Rotation augmentation
        translate=0.1,          # Translation augmentation
        scale=0.5,              # Scale augmentation
        shear=2.0,              # Shear augmentation
        perspective=0.0001,     # Perspective augmentation
        close_mosaic=10,        # Disable mosaic last 10 epochs

        # Training settings
        weight_decay=0.0005,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.1,    # Helps with noisy labels

        # Save settings
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        project="runs/retrain",
        name="ppe_best",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    return results


def deploy_model():
    """Copy the best model to the models/ directory"""
    # Find the best model from retraining
    best_model = Path("runs/retrain/ppe_best/weights/best.pt")

    if best_model.exists():
        # Backup old model
        old_model = Path("models/best.pt")
        if old_model.exists():
            backup = Path("models/best_old_backup.pt")
            shutil.copy2(old_model, backup)
            print(f"\n📦 Old model backed up to: {backup}")

        # Copy new model
        shutil.copy2(best_model, old_model)
        print(f"✅ New model deployed to: {old_model}")

        # Show model info
        from ultralytics import YOLO
        model = YOLO(str(old_model))
        print(f"\n📊 New model info:")
        print(f"   Classes: {model.names}")
        model.info()
    else:
        print(f"❌ Best model not found at {best_model}")
        print("   Check runs/retrain/ppe_best/weights/ for available models")


def validate_model():
    """Validate the newly trained model"""
    from ultralytics import YOLO

    model = YOLO("models/best.pt")
    print(f"\n{'=' * 60}")
    print("📊 Validating new model...")
    print(f"{'=' * 60}")

    # Check if validation data exists
    data_yaml = "./dataset/data.yaml"
    if os.path.exists(data_yaml):
        metrics = model.val(data=data_yaml, imgsz=640, conf=0.25)
        print(f"\n✅ Validation Results:")
        print(f"   mAP50:    {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   Per-class AP50:")
        for i, name in model.names.items():
            if i < len(metrics.box.ap50):
                print(f"     {name}: {metrics.box.ap50[i]:.4f}")
    else:
        print(f"⚠️  Dataset not found at {data_yaml}, skipping validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain PPE Detection Model")
    parser.add_argument("--api-key", type=str, required=True,
                        help="Roboflow API key (get from https://app.roboflow.com/settings/api)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--model-size", type=str, default="s",
                        choices=["n", "s", "m", "l"],
                        help="YOLOv8 model size: n=nano, s=small, m=medium, l=large (default: s)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (reduce if running out of memory, e.g. 8)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (if already downloaded)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (just deploy and validate)")

    args = parser.parse_args()

    try:
        # Step 1: Download dataset
        if not args.skip_download:
            data_yaml = download_dataset(args.api_key)
        else:
            data_yaml = "./dataset/data.yaml"
            if not os.path.exists(data_yaml):
                print(f"❌ Dataset not found at {data_yaml}")
                print("   Remove --skip-download flag to download it")
                exit(1)
            print(f"⏭️  Skipping download, using existing dataset at {data_yaml}")

        # Step 2: Train
        if not args.skip_train:
            train_model(
                data_yaml=data_yaml,
                epochs=args.epochs,
                model_size=args.model_size,
                batch=args.batch,
                imgsz=args.imgsz,
            )

        # Step 3: Deploy
        deploy_model()

        # Step 4: Validate
        validate_model()

        print(f"\n{'=' * 60}")
        print("🎉 DONE! Your model has been retrained and deployed.")
        print("   Restart the Flask app to use the new model.")
        print(f"{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted. Partial results may be in runs/retrain/")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
