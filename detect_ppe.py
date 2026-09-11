from ultralytics import YOLO
import os
import sys

def detect_image():
    """Detect PPE from image"""
    image_path = input("Enter image path (e.g., source_files/image.jpg): ").strip()
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print(f"\nDetecting PPE in image: {image_path}")
    model = YOLO("models/best.pt")
    results = model.predict(source=image_path, save=True, conf=0.5)
    print("✓ Detection complete! Results saved in 'runs/detect/predict/' folder")

def detect_video():
    """Detect PPE from video"""
    video_path = input("Enter video path (e.g., source_files/video.mp4): ").strip()
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    
    print(f"\nDetecting PPE in video: {video_path}")
    model = YOLO("models/best.pt")
    results = model.predict(source=video_path, save=True, conf=0.5)
    print("✓ Detection complete! Video saved in 'runs/detect/predict/' folder")

def detect_webcam():
    """Detect PPE from webcam"""
    print("\nStarting webcam detection... (Press 'q' to quit)")
    model = YOLO("models/best.pt")
    results = model.predict(source=0, conf=0.5)  # 0 = webcam
    print("✓ Webcam detection stopped!")

def main():
    print("=" * 50)
    print("   PPE DETECTION FOR CONSTRUCTION SITE SAFETY")
    print("=" * 50)
    print("\nSelect detection mode:")
    print("1. Image")
    print("2. Video")
    print("3. Webcam")
    print("4. Exit")
    print("-" * 50)
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            detect_image()
        elif choice == "2":
            detect_video()
        elif choice == "3":
            detect_webcam()
        elif choice == "4":
            print("Exiting... Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice! Please enter 1, 2, 3, or 4")
        
        print("\n" + "-" * 50)
        cont = input("Run another detection? (y/n): ").strip().lower()
        if cont != 'y':
            print("Thank you for using PPE Detection!")
            break

if __name__ == "__main__":
    main()
