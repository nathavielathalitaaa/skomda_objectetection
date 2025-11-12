"""
DTP AI Project - Object Detector
Flask Web Application untuk ROI YOLO Detection dengan Face Detection
Aplikasi web interaktif dan elegan untuk deteksi objek real-time dengan ROI
Created by: SMK Telkom Sidoarjo - DTP AI Specialist
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
from threading import Lock
import os
from werkzeug.utils import secure_filename

# DeepFace disabled temporarily due to compatibility issues
# Will use basic face detection only
DEEPFACE_AVAILABLE = False
print("ℹ️ Using basic face detection (DeepFace disabled for compatibility)")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
camera = None
video_capture = None
model = None
face_cascade = None
emotion_detection_enabled = True
detection_active = False
camera_active = False
video_mode = False  # False = camera, True = video file
current_video_path = None
camera_lock = Lock()

# Emotion labels (Indonesian)
EMOTIONS = {
    'angry': 'Marah',
    'disgust': 'Jijik',
    'fear': 'Takut',
    'happy': 'Senang',
    'sad': 'Sedih',
    'surprise': 'Terkejut',
    'neutral': 'Netral'
}

# Settings
settings = {
    'roi_type': 'center',
    'conf_threshold': 0.5,
    'roi_coords': None,
    'show_roi': True,
    'show_labels': True,
    'save_video': False,
    'detect_emotion': True
}

# Statistics
stats = {
    'objects_in_roi': 0,
    'objects_outside_roi': 0,
    'total_detections': 0,
    'fps': 0,
    'detected_objects': {},
    'source': 'camera',
    'detected_emotions': {}
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load YOLO model"""
    global model, face_cascade
    if model is None:
        print("Loading YOLO model...")
        model = YOLO('yolov8n.pt')
        print("Model loaded successfully!")
    
    # Load face cascade for face detection
    if face_cascade is None:
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("Face cascade loaded successfully!")
        except Exception as e:
            print(f"Error loading face cascade: {e}")
            face_cascade = None
    
    return model

def define_roi(image_shape, roi_type='center', custom_roi=None):
    """Define Region of Interest"""
    h, w = image_shape[:2]
    
    if roi_type == 'center':
        x1, y1 = int(w * 0.25), int(h * 0.25)
        x2, y2 = int(w * 0.75), int(h * 0.75)
    elif roi_type == 'top':
        x1, y1 = 0, 0
        x2, y2 = w, int(h * 0.5)
    elif roi_type == 'bottom':
        x1, y1 = 0, int(h * 0.5)
        x2, y2 = w, h
    elif roi_type == 'left':
        x1, y1 = 0, 0
        x2, y2 = int(w * 0.5), h
    elif roi_type == 'right':
        x1, y1 = int(w * 0.5), 0
        x2, y2 = w, h
    elif roi_type == 'custom' and custom_roi:
        x1, y1, x2, y2 = custom_roi
    else:
        x1, y1, x2, y2 = 0, 0, w, h
    
    return (x1, y1, x2, y2)

def detect_emotion(face_img):
    """Detect emotion from face image - PLACEHOLDER"""
    # DeepFace disabled for compatibility
    # Return None for now
    return None

def is_in_roi(box, roi):
    """Check if bounding box is inside ROI"""
    box_x1, box_y1, box_x2, box_y2 = box
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    
    center_x = (box_x1 + box_x2) / 2
    center_y = (box_y1 + box_y2) / 2
    
    return (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2)

def generate_frames():
    """Generate video frames with object detection"""
    global camera, video_capture, detection_active, stats, video_mode, current_video_path, camera_active
    
    # Load model
    model = load_model()
    
    # Initialize video source
    if video_mode and current_video_path:
        # Video file mode
        video_capture = cv2.VideoCapture(current_video_path)
        stats['source'] = f'video: {os.path.basename(current_video_path)}'
        cap = video_capture
    else:
        # Camera mode
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        stats['source'] = 'camera'
        camera_active = True
        cap = camera
    
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    detection_active = True
    
    while detection_active:
        success, frame = cap.read()
        if not success:
            # If video ended, loop it
            if video_mode and current_video_path:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        fps_counter += 1
        if fps_counter >= 30:
            current_fps = fps_counter / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0
            stats['fps'] = current_fps
        
        h, w = frame.shape[:2]
        
        # Define ROI
        roi = define_roi((h, w), settings['roi_type'], settings['roi_coords'])
        roi_x1, roi_y1, roi_x2, roi_y2 = roi
        
        # Detect objects
        results = model(frame, conf=settings['conf_threshold'], verbose=False)
        
        count_in_roi = 0
        count_outside_roi = 0
        detected_objects = {}
        detected_emotions = {}
        
        # Detect faces and emotions (only every 5 frames for performance)
        if settings.get('detect_emotion', True) and DEEPFACE_AVAILABLE and face_cascade is not None and fps_counter % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (fx, fy, fw, fh) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 255), 2)
                
                # Extract face ROI for emotion detection
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                
                # Detect emotion
                try:
                    emotion = detect_emotion(face_roi)
                    if emotion:
                        detected_emotions[emotion] = detected_emotions.get(emotion, 0) + 1
                        
                        # Draw emotion label
                        cv2.putText(frame, f'Ekspresi: {emotion}', (fx, fy - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                except:
                    pass
        elif settings.get('detect_emotion', True) and face_cascade is not None and fps_counter % 10 == 0:
            # Jika DeepFace tidak available, hanya deteksi wajah tanpa emosi
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (fx, fy, fw, fh) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 255), 2)
                cv2.putText(frame, 'Wajah Terdeteksi', (fx, fy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                
                in_roi = is_in_roi((x1, y1, x2, y2), roi)
                
                if in_roi:
                    color = (0, 255, 0)  # Green
                    count_in_roi += 1
                    detected_objects[label] = detected_objects.get(label, 0) + 1
                else:
                    color = (0, 165, 255)  # Orange
                    count_outside_roi += 1
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                if settings['show_labels']:
                    text = f'{label} {conf:.2f}'
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Update stats
        stats['objects_in_roi'] = count_in_roi
        stats['objects_outside_roi'] = count_outside_roi
        stats['total_detections'] = count_in_roi + count_outside_roi
        stats['detected_objects'] = detected_objects
        stats['detected_emotions'] = detected_emotions
        
        # Draw ROI
        if settings['show_roi']:
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 3)
            cv2.putText(frame, 'ROI', (roi_x1 + 10, roi_y1 + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Draw info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        info_y = 35
        cv2.putText(frame, f'FPS: {current_fps:.1f}', (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 30
        cv2.putText(frame, f'Objects in ROI: {count_in_roi}', (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        info_y += 30
        cv2.putText(frame, f'Objects outside: {count_outside_roi}', (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        info_y += 30
        cv2.putText(frame, f'Confidence: {settings["conf_threshold"]:.2f}', (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Cleanup
    if camera is not None:
        camera.release()
        camera_active = False
    if video_capture is not None:
        video_capture.release()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Get current statistics"""
    return jsonify(stats)

@app.route('/settings', methods=['GET', 'POST'])
def update_settings():
    """Update detection settings"""
    global settings
    
    if request.method == 'POST':
        data = request.json
        
        if 'roi_type' in data:
            settings['roi_type'] = data['roi_type']
        
        if 'conf_threshold' in data:
            settings['conf_threshold'] = float(data['conf_threshold'])
        
        if 'show_roi' in data:
            settings['show_roi'] = bool(data['show_roi'])
        
        if 'show_labels' in data:
            settings['show_labels'] = bool(data['show_labels'])
        
        if 'detect_emotion' in data:
            settings['detect_emotion'] = bool(data['detect_emotion'])
        
        if 'roi_coords' in data:
            settings['roi_coords'] = data['roi_coords']
        
        return jsonify({'status': 'success', 'settings': settings})
    
    return jsonify(settings)

@app.route('/stop')
def stop_detection():
    """Stop detection"""
    global detection_active, camera, video_capture, camera_active
    detection_active = False
    camera_active = False
    if camera is not None:
        camera.release()
        camera = None
    if video_capture is not None:
        video_capture.release()
        video_capture = None
    return jsonify({'status': 'stopped'})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload video file for detection"""
    global video_mode, current_video_path, detection_active
    
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Stop current detection
        detection_active = False
        time.sleep(0.5)
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Set video mode
        video_mode = True
        current_video_path = filepath
        
        return jsonify({
            'status': 'success',
            'message': f'Video {filename} uploaded successfully',
            'filename': filename
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wmv, flv, webm'
        }), 400

@app.route('/switch_to_camera', methods=['POST'])
def switch_to_camera():
    """Switch back to camera mode"""
    global video_mode, current_video_path, detection_active
    
    # Stop current detection
    detection_active = False
    time.sleep(0.5)
    
    # Switch to camera
    video_mode = False
    current_video_path = None
    
    return jsonify({
        'status': 'success',
        'message': 'Switched to camera mode'
    })

@app.route('/status')
def get_status():
    """Get current application status"""
    return jsonify({
        'detection_active': detection_active,
        'camera_active': camera_active,
        'video_mode': video_mode,
        'current_video': os.path.basename(current_video_path) if current_video_path else None
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 DTP AI Project - Object Detector")
    print("   SMK Telkom Sidoarjo - DTP AI Specialist")
    print("=" * 70)
    print("📡 Server starting on http://localhost:5000")
    print("🎥 Webcam Detection Ready")
    print("🎯 YOLO Object Detection with ROI")
    if DEEPFACE_AVAILABLE:
        print("😊 Face Emotion Recognition ENABLED")
    else:
        print("😊 Face Detection ONLY (Emotion recognition disabled)")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
