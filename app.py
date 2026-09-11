from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from ultralytics import YOLO
import os
import cv2
import base64
import numpy as np
from werkzeug.utils import secure_filename
import shutil
from pathlib import Path
import glob

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'avi', 'mov', 'mkv'}

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# Load model
try:
    model = YOLO('models/best.pt')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_ppe(file_path, is_video=False, conf=0.25):
    """Run PPE detection on image or video"""
    try:
        if model is None:
            return None, False, "Model not loaded", {}
        
        print(f"Starting detection on: {file_path} (conf={conf})")
        # augment=True enables test-time augmentation (TTA) for better accuracy
        # imgsz=1280 gives the model more detail to work with
        # iou=0.5 allows more overlapping detections through NMS
        results = model.predict(
            source=file_path,
            save=True,
            conf=conf,
            augment=True,
            imgsz=1280,
            iou=0.5
        )
        
        print(f"Detection complete. {len(results)} frame(s) processed.")
        
        # --- Aggregate detection counts ---
        frames_analyzed = len(results)
        
        # Collect raw per-frame counts
        per_frame_counts = []
        for i, r in enumerate(results):
            frame_counts = {}
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    c = float(box.conf[0])
                    frame_counts[label] = frame_counts.get(label, 0) + 1
                # Log first 5 frames and every 50th frame for debugging
                if i < 5 or i % 50 == 0:
                    print(f"  Frame {i}: {frame_counts}")
            per_frame_counts.append(frame_counts)
        
        # Collect all classes seen across all frames
        all_classes = set()
        for fc in per_frame_counts:
            all_classes.update(fc.keys())
        
        # Count how many frames each class appears in
        class_frame_presence = {}
        for cls in all_classes:
            class_frame_presence[cls] = sum(1 for fc in per_frame_counts if fc.get(cls, 0) > 0)
        
        # --- Build PPE compliance status (True/False for each PPE item) ---
        # ALWAYS show all 3 items so the user sees the full safety picture
        # Logic: if positive class detected → True
        #        if only negative class detected or neither detected → False
        ppe_items = [
            {'name': 'Hardhat', 'positive': 'Hardhat', 'negative': 'NO-Hardhat'},
            {'name': 'Mask', 'positive': 'Mask', 'negative': 'NO-Mask'},
            {'name': 'Safety Vest', 'positive': 'Safety Vest', 'negative': 'NO-Safety Vest'},
        ]
        
        ppe_status = []
        for item in ppe_items:
            pos_frames = class_frame_presence.get(item['positive'], 0)
            neg_frames = class_frame_presence.get(item['negative'], 0)
            
            # Detected = True only if positive class appears in more frames
            detected = pos_frames > 0 and pos_frames >= neg_frames
            
            ppe_status.append({
                'name': item['name'],
                'detected': detected,
                'positive_frames': pos_frames,
                'negative_frames': neg_frames
            })
        
        # --- Count other objects (Person, Safety Cone, machinery, vehicle) ---
        other_classes = ['Person', 'Safety Cone', 'machinery', 'vehicle']
        from collections import Counter
        other_counts = {}
        for cls in other_classes:
            if cls in all_classes:
                if is_video and frames_analyzed > 1:
                    # Use MODE for video
                    nonzero = [fc.get(cls, 0) for fc in per_frame_counts if fc.get(cls, 0) > 0]
                    if nonzero:
                        counter = Counter(nonzero)
                        other_counts[cls] = counter.most_common(1)[0][0]
                else:
                    # Direct count for images
                    other_counts[cls] = sum(fc.get(cls, 0) for fc in per_frame_counts)
        
        print(f"PPE Status: {ppe_status}")
        print(f"Other counts: {other_counts} (frames: {frames_analyzed})")
        
        # Pack extra info for the response
        detection_info = {
            'ppe_status': ppe_status,
            'other_counts': other_counts,
            'frames_analyzed': frames_analyzed,
            'is_video': is_video
        }
        
        # Check if results were saved in runs/detect
        detect_dir = Path('runs/detect')
        if detect_dir.exists():
            # Find all predict folders and get the latest
            predict_dirs = list(detect_dir.glob('predict*'))
            if predict_dirs:
                latest = max(predict_dirs, key=lambda p: p.stat().st_mtime)
                print(f"Found latest detection folder: {latest}")
                
                # Find all image/video files in the latest folder (exclude .txt, .npy)
                result_files = [f for f in latest.glob('*') if f.suffix.lower() in {'.jpg', '.png', '.jpeg', '.mp4', '.avi', '.mov'}]
                
                if result_files:
                    result_file = result_files[0]
                    print(f"Copying result from: {result_file}")
                    
                    filename = f"{Path(file_path).stem}_detected{result_file.suffix}"
                    dest = os.path.join(app.config['RESULTS_FOLDER'], filename)
                    shutil.copy2(result_file, dest)
                    print(f"Result saved to: {dest}")
                    return filename, True, "Detection successful!", detection_info
        
        return None, False, "No detection results found", detection_info
    except Exception as e:
        print(f"❌ Error in detection: {e}")
        import traceback
        traceback.print_exc()
        return None, False, str(e), {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("\n📤 Upload request received")
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    file_type = request.form.get('type', 'image')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use jpg, png, mp4, avi, etc.'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✓ File saved: {filepath}")
        
        # Run detection
        conf = float(request.form.get('conf', 0.25))
        result_filename, success, message, detection_info = detect_ppe(filepath, is_video=(file_type == 'video'), conf=conf)
        
        # Clean up upload
        try:
            os.remove(filepath)
            print(f"✓ Cleaned up upload file")
        except:
            pass
        
        if success:
            print(f"✓ Detection succeeded: {result_filename}")
            return jsonify({
                'success': True,
                'message': message,
                'result_file': result_filename,
                'preview_url': f'/preview/{result_filename}',
                'ppe_status': detection_info.get('ppe_status', []),
                'other_counts': detection_info.get('other_counts', {}),
                'frames_analyzed': detection_info.get('frames_analyzed', 1),
                'is_video': detection_info.get('is_video', False)
            })
        else:
            print(f"❌ Detection failed: {message}")
            return jsonify({
                'success': False,
                'error': message
            }), 500
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/webcam-frame', methods=['POST'])
def webcam_frame():
    """Run detection on a single webcam frame sent as base64 from the browser"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode base64 image
        img_data = data['image'].split(',')[1]  # remove data:image/...;base64, prefix
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # Run YOLO detection with TTA for better accuracy
        conf = data.get('conf', 0.25)
        results = model.predict(source=frame, save=False, conf=conf, augment=True, iou=0.5)

        # Draw detections on the frame
        annotated = results[0].plot()

        # Encode back to base64 JPEG
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        result_b64 = base64.b64encode(buffer).decode('utf-8')

        # Gather detection info
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = results[0].names[cls_id]
                detections.append({'label': label, 'confidence': round(conf, 2)})

        return jsonify({
            'success': True,
            'image': 'data:image/jpeg;base64,' + result_b64,
            'detections': detections
        })
    except Exception as e:
        print(f"❌ Webcam frame error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/preview/<filename>')
def preview_file(filename):
    filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'File not found'}), 404

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete a detection result file"""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], secure_filename(filename))
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✓ Deleted result: {filename}")
            return jsonify({'success': True, 'message': f'Deleted {filename}'})
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(f"❌ Delete error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/results')
def get_results():
    """Get list of detection results"""
    results = []
    try:
        if os.path.exists(app.config['RESULTS_FOLDER']):
            for file in sorted(os.listdir(app.config['RESULTS_FOLDER']), reverse=True):
                filepath = os.path.join(app.config['RESULTS_FOLDER'], file)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
                    results.append({
                        'filename': file,
                        'size': f'{size:.2f} MB',
                        'type': 'image' if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')) else 'video',
                        'preview_url': f'/preview/{file}'
                    })
    except Exception as e:
        print(f"Error loading results: {e}")
    
    return jsonify(results)

@app.route('/clear-all', methods=['POST'])
def clear_all():
    """Clear all uploads and detection results from the server"""
    cleared = {'uploads': 0, 'results': 0}
    try:
        # Clear uploads folder
        upload_dir = app.config['UPLOAD_FOLDER']
        if os.path.exists(upload_dir):
            for f in os.listdir(upload_dir):
                fp = os.path.join(upload_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
                    cleared['uploads'] += 1

        # Clear results folder
        results_dir = app.config['RESULTS_FOLDER']
        if os.path.exists(results_dir):
            for f in os.listdir(results_dir):
                fp = os.path.join(results_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
                    cleared['results'] += 1

        # Clear any leftover YOLO prediction folders
        runs_dir = 'runs/detect'
        if os.path.exists(runs_dir):
            shutil.rmtree(runs_dir, ignore_errors=True)
            os.makedirs(runs_dir, exist_ok=True)

        print(f"🧹 Cleared {cleared['uploads']} uploads + {cleared['results']} results")
        return jsonify({'success': True, 'cleared': cleared})
    except Exception as e:
        print(f"❌ Clear error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Starting PPE Detection Web App")
    print("=" * 50)
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
