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
import torch
import time
import json
from threading import Lock, Thread
import os
from werkzeug.utils import secure_filename
import csv
from datetime import datetime
import winsound

# DeepFace disabled temporarily due to compatibility issues
# Will use basic face detection only
DEEPFACE_AVAILABLE = False
print("ℹ️ Using basic face detection (DeepFace disabled for compatibility)")

# Database paths
DATASET_DIR = 'datasets'
FACES_DIR = os.path.join(DATASET_DIR, 'faces')
OBJECTS_DIR = os.path.join(DATASET_DIR, 'objects')
FACE_DB_CSV = os.path.join(DATASET_DIR, 'database_ekspresi_terdeteksi.csv')
OBJECT_DB_CSV = os.path.join(DATASET_DIR, 'database_objek_terdeteksi.csv')

# Create directories
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(OBJECTS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

# Create upload folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize CSV databases
def init_databases():
    """Initialize CSV database files with headers"""
    if not os.path.exists(FACE_DB_CSV):
        with open(FACE_DB_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Waktu', 'Tanggal', 'Jam', 'Latar Tempat', 'Objek Terdeteksi', 
                           'Ekspresi', 'Kelamin', 'Keterangan', 'Foto'])
        print(f"✅ Face DB created: {FACE_DB_CSV}")
    
    if not os.path.exists(OBJECT_DB_CSV):
        with open(OBJECT_DB_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Waktu', 'Tanggal', 'Jam', 'Latar Tempat', 'Objek Terdeteksi', 
                           'Jumlah', 'Confidence', 'Lokasi ROI', 'Foto'])
        print(f"✅ Object DB created: {OBJECT_DB_CSV}")

init_databases()

# Face tracking
known_faces = []
person_counter = 0
last_save_time = {}
SAVE_COOLDOWN = 5  # seconds

# Global variables
camera = None
video_capture = None
model = None
face_cascade = None
emotion_detection_enabled = True
# Enable saving/logging by default so new users don't need to press a toggle
detection_active = True
save_enabled = False  # Only save/beep when on detector page
camera_active = False
video_mode = False  # False = camera, True = video file
current_video_path = None
# Additional source modes
ip_mode = False  # True when using IP camera URL
ip_camera_url = None
camera_index = 0  # default webcam index
camera_lock = Lock()

# Streaming buffer (decouples camera loop from HTTP)
latest_frame_bytes = None
frame_lock = Lock()  # Protect latest_frame_bytes from race conditions
capture_thread = None
capture_running = False

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
    'detect_emotion': True,
    'flip_horizontal': True,
    # Performance
    'device': 'auto',           # 'auto' | 'cpu' | 'cuda'
    'use_half': True,           # FP16 when on CUDA
    'imgsz': 640,               # inference image size (back to normal quality)
    'detect_interval': 2,       # run detection every N frames (2 for balance)
    # Save behavior
    'single_save_mode': True    # save only once per unique object label / face id per session
}

# Diagnostic flag for frame issues
DEBUG_FRAME_LOG = True

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

# Session memory for single-save mode
saved_object_labels = set()
saved_face_ids = set()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def play_beep():
    """Play beep sound"""
    try:
        winsound.Beep(1000, 200)
    except:
        pass

def extract_face_features(face_img):
    """Extract face histogram"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_faces(face_hist, known_face_hists, threshold=0.7):
    """Compare face with known faces"""
    for idx, known_hist in enumerate(known_face_hists):
        correlation = cv2.compareHist(face_hist, known_hist, cv2.HISTCMP_CORREL)
        if correlation > threshold:
            return True, idx + 1
    return False, -1

def save_face_to_db(face_img, person_id, expression="Netral", location="Camera"):
    """Save face to database"""
    global save_enabled
    if not save_enabled:
        return None  # Skip saving if not on detector page
    
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Ensure per-person folder
    person_folder = os.path.join(FACES_DIR, f"person_{person_id}")
    os.makedirs(person_folder, exist_ok=True)
    filename = f"person_{person_id}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(person_folder, filename)
    cv2.imwrite(filepath, face_img)
    
    with open(FACE_DB_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, date_str, time_str, location, "Wajah Manusia", 
                        expression, "Tidak Diketahui", f"Person {person_id}", os.path.relpath(filepath, DATASET_DIR)])
    
    play_beep()
    print(f"🔔 Face saved: Person {person_id}")
    return filepath

def save_object_detection(frame, label, bbox, confidence, in_roi, location="Webcam"):
    """Save cropped object image under datasets/objects/<label>/ and log to CSV."""
    global save_enabled
    if not save_enabled:
        return None  # Skip saving if not on detector page
    
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    now = datetime.now()
    label_folder = os.path.join(OBJECTS_DIR, label)
    os.makedirs(label_folder, exist_ok=True)
    filename = f"{label}_{now.strftime('%Y%m%d_%H%M%S')}_{x1}-{y1}-{x2}-{y2}.jpg"
    filepath = os.path.join(label_folder, filename)
    cv2.imwrite(filepath, crop)
    # CSV row
    roi_location = "Inside ROI" if in_roi else "Outside ROI"
    with open(OBJECT_DB_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            now.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            location,
            label,
            1,
            f"{confidence:.2f}",
            roi_location,
            os.path.relpath(filepath, DATASET_DIR)
        ])
    play_beep()
    print(f"🔔 Object saved: {label} -> {filepath}")
    return filepath

def save_object_to_db(object_name, count, confidence, in_roi, location="Camera"):
    """Save object to database"""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    roi_location = "Inside ROI" if in_roi else "Outside ROI"
    
    with open(OBJECT_DB_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, date_str, time_str, location, object_name, 
                        count, f"{confidence:.2f}", roi_location, "-"])
    
    play_beep()
    print(f"🔔 Object saved: {object_name}")

def load_model():
    """Load YOLO model"""
    global model, face_cascade
    if model is None:
        print("Loading YOLO model...")
        model = YOLO('models/yolov8n.pt')  # Updated path
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

def _capture_loop():
    """Background capture loop: grabs frames and updates latest_frame_bytes."""
    global camera, video_capture, detection_active, stats, video_mode, current_video_path
    global camera_active, latest_frame_bytes, capture_running, frame_lock
    
    print("🎥 Starting background capture loop...")
    model = load_model()
    # Resolve device and precision
    try:
        if settings.get('device', 'auto') == 'cuda' and torch.cuda.is_available():
            device_arg = 0
        elif settings.get('device', 'auto') == 'cpu':
            device_arg = 'cpu'
        else:
            device_arg = 0 if torch.cuda.is_available() else 'cpu'
        use_half = bool(settings.get('use_half', True)) and (device_arg != 'cpu') and torch.cuda.is_available()
        imgsz = int(settings.get('imgsz', 416))  # Reduced from 640 to 416 for speed
        detect_interval = max(1, int(settings.get('detect_interval', 3)))  # Increased to 3 frames for better FPS
    except Exception:
        device_arg = 'cpu'
        use_half = False
        imgsz = 416
        detect_interval = 3
    
    # Open source
    if video_mode and current_video_path:
        cap = cv2.VideoCapture(current_video_path)
        stats['source'] = f'video: {os.path.basename(current_video_path)}'
    elif ip_mode and ip_camera_url:
        cap = cv2.VideoCapture(ip_camera_url)
        stats['source'] = f'ip: {ip_camera_url}'
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Optimized camera settings for maximum performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG for faster capture
            stats['source'] = f'camera:{camera_index}'
            camera_active = True
            print(f"✅ Opened camera index {camera_index}")
            # Warm up camera - discard first frames
            for _ in range(3):  # Reduced from 5 to 3
                cap.read()
        else:
            print(f"❌ Failed to open camera index {camera_index}. Will stream black frames.")
            stats['source'] = f'camera-failed:{camera_index}'
    
    fps_start_time = time.time()
    fps_counter = 0
    frame_index = 0
    last_detections = []  # cache of last detections for drawing during skip frames
    
    try:
        while capture_running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            
            # Un-mirror camera if enabled (apply only for live camera by default)
            if settings.get('flip_horizontal', False) and not video_mode and not ip_mode:
                frame = cv2.flip(frame, 1)
            
            # Update FPS less frequently to reduce overhead
            fps_counter += 1
            if fps_counter >= 30:  # Changed from 15 to 30 frames
                stats['fps'] = fps_counter / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_counter = 0
            frame_index += 1

            # Run detection (optionally skip frames to improve FPS)
            count_in_roi = 0
            count_outside_roi = 0
            detected_objects = {}
            try:
                h, w = frame.shape[:2]
                roi_x1, roi_y1, roi_x2, roi_y2 = define_roi((h, w), settings['roi_type'], settings['roi_coords'])
                roi = (roi_x1, roi_y1, roi_x2, roi_y2)

                run_inference = (frame_index % detect_interval == 0)
                if run_inference:
                    results = model(frame, conf=settings['conf_threshold'], verbose=False,
                                    device=device_arg, half=use_half, imgsz=imgsz)
                    # Convert detections into a simple cache structure
                    detections_cache = []
                    for r in results:
                        for b in r.boxes:
                            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            conf = float(b.conf[0])
                            cls = int(b.cls[0])
                            label = model.names.get(cls, str(cls))
                            detections_cache.append((x1, y1, x2, y2, conf, label))
                    last_detections = detections_cache
                else:
                    # Use last detections to avoid flicker and keep UI responsive
                    results = None

                # Iterate over current cache
                for (x1, y1, x2, y2, conf, label) in list(last_detections):

                        in_roi = is_in_roi((x1, y1, x2, y2), roi)
                        color = (0,255,0) if in_roi else (0,165,255)
                        if in_roi:
                            count_in_roi += 1
                        else:
                            count_outside_roi += 1
                        detected_objects[label] = detected_objects.get(label, 0) + 1

                        # Draw box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        if settings.get('show_labels', True):
                            text = f"{label} {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
                            cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

                        # Save only on inference frames
                        if run_inference:
                            try:
                                if settings.get('single_save_mode', True):
                                    # Save only once per unique label per session
                                    if label not in saved_object_labels:
                                        save_object_detection(frame, label, (x1, y1, x2, y2), conf, in_roi, location="Webcam")
                                        saved_object_labels.add(label)
                                else:
                                    now_ts = time.time()
                                    key = f"obj_{label}_{'in' if in_roi else 'out'}"
                                    if key not in last_save_time or (now_ts - last_save_time[key]) > (SAVE_COOLDOWN if in_roi else SAVE_COOLDOWN*2):
                                        save_object_detection(frame, label, (x1, y1, x2, y2), conf, in_roi, location="Webcam")
                                        last_save_time[key] = now_ts
                            except Exception:
                                pass

                # Face detection (reduced frequency for performance - every 20 frames)
                if face_cascade is not None and (frame_index % 20 == 0):
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))  # Stricter parameters
                    for (fx, fy, fw, fh) in faces:
                        face_roi = frame[fy:fy+fh, fx:fx+fw]
                        # Identify simple by histogram
                        face_hist = extract_face_features(face_roi)
                        is_known, person_id = compare_faces(face_hist, known_faces, threshold=0.7)
                        if not is_known:
                            known_faces.append(face_hist)
                            person_id = len(known_faces)
                        # Save per person: single once per session or cooldown
                        if settings.get('single_save_mode', True):
                            if person_id not in saved_face_ids:
                                save_face_to_db(face_roi, person_id, expression="Netral", location="Webcam")
                                saved_face_ids.add(person_id)
                        else:
                            kf = f"face_{person_id}"
                            now_ts = time.time()
                            if kf not in last_save_time or (now_ts - last_save_time[kf]) > SAVE_COOLDOWN:
                                save_face_to_db(face_roi, person_id, expression="Netral", location="Webcam")
                                last_save_time[kf] = now_ts
                        # Draw rectangle and label
                        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 255), 2)
                        cv2.putText(frame, f"Person {person_id}", (fx, fy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)

                # Draw ROI if enabled
                if settings.get('show_roi', True):
                    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,0,255), 2)

                # Update stats
                stats['objects_in_roi'] = count_in_roi
                stats['objects_outside_roi'] = count_outside_roi
                stats['total_detections'] = count_in_roi + count_outside_roi
                stats['detected_objects'] = detected_objects
            except Exception:
                # Keep streaming even if detection errors occur
                pass
            
            # Encode frame to JPEG - optimized for speed
            ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                # Diagnostics: check brightness and shape
                try:
                    mean_brightness = float(np.mean(frame))
                    h, w = frame.shape[:2]
                except Exception:
                    mean_brightness = -1
                    h, w = (0, 0)

                if DEBUG_FRAME_LOG and mean_brightness >= 0:
                    # Only log when unusually dark or periodically
                    if mean_brightness < 20 or frame_index % 200 == 0:
                        print(f"[FRAME-DBG] idx={frame_index} mean_brightness={mean_brightness:.2f} shape={w}x{h}")

                with frame_lock:
                    latest_frame_bytes = buf.tobytes()
    finally:
        cap.release()
        camera_active = False
        print("🛑 Background capture loop stopped")


def ensure_capture_running():
    """Start background capture if not already running."""
    global capture_thread, capture_running
    if capture_thread is None or not capture_thread.is_alive():
        capture_running = True
        capture_thread = Thread(target=_capture_loop, daemon=True)
        capture_thread.start()

def stop_capture():
    """Stop background capture thread and reset streaming buffer."""
    global capture_running, capture_thread, latest_frame_bytes, camera_active
    capture_running = False
    # Give loop a moment to exit and release camera
    try:
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=1.0)
    except Exception:
        pass
    with frame_lock:
        latest_frame_bytes = None
    camera_active = False

def restart_capture():
    """Restart capture with current source settings."""
    stop_capture()
    # small delay to ensure camera release
    time.sleep(0.2)
    ensure_capture_running()


def generate_frames():
    """MJPEG generator that serves the latest frame from background loop."""
    ensure_capture_running()
    # Wait until first frame is available
    start_wait = time.time()
    while latest_frame_bytes is None and time.time() - start_wait < 3:
        time.sleep(0.05)
    
    while True:
        with frame_lock:
            current_frame = latest_frame_bytes
        
        if current_frame is None:
            # Send black frame placeholder
            blk = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buf = cv2.imencode('.jpg', blk)
            frame = buf.tobytes()
            if not capture_running:
                # No capture running and no frame: end stream
                break
        else:
            frame = current_frame
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.025)  # ~40 FPS limit for smoother streaming

@app.route('/')
def landing():
    """Landing page"""
    return render_template('landing.html')

@app.route('/detector')
def detector():
    """Object detector page"""
    return render_template('index.html')

@app.route('/gallery')
def gallery_page():
    """Gallery page"""
    return render_template('gallery.html')

@app.route('/test_camera')
def test_camera():
    """Test if camera can be opened"""
    try:
        test_cam = cv2.VideoCapture(0)
        if test_cam.isOpened():
            ret, frame = test_cam.read()
            test_cam.release()
            if ret:
                return jsonify({'status': 'success', 'message': 'Camera working!'})
            else:
                return jsonify({'status': 'error', 'message': 'Camera opened but cannot read frame'})
        else:
            return jsonify({'status': 'error', 'message': 'Cannot open camera'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    print("📡 /video_feed endpoint called")
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
        
        if 'flip_horizontal' in data:
            settings['flip_horizontal'] = bool(data['flip_horizontal'])
        
        if 'roi_coords' in data:
            settings['roi_coords'] = data['roi_coords']

        # Performance settings updates
        if 'device' in data:
            if data['device'] in ('auto', 'cpu', 'cuda'):
                settings['device'] = data['device']
        if 'use_half' in data:
            settings['use_half'] = bool(data['use_half'])
        if 'imgsz' in data:
            try:
                settings['imgsz'] = max(64, int(data['imgsz']))
            except Exception:
                pass
        if 'detect_interval' in data:
            try:
                settings['detect_interval'] = max(1, int(data['detect_interval']))
            except Exception:
                pass
        if 'single_save_mode' in data:
            settings['single_save_mode'] = bool(data['single_save_mode'])
        
        return jsonify({'status': 'success', 'settings': settings})
    
    return jsonify(settings)

@app.route('/start', methods=['POST'])
def start_detection():
    """Start detection"""
    global detection_active, video_mode, current_video_path, ip_mode
    # Start live camera by default when starting from UI toggle
    video_mode = False
    current_video_path = None
    ip_mode = False
    detection_active = True
    restart_capture()
    return jsonify({'status': 'started', 'message': 'Detection started successfully'})

@app.route('/stop')
def stop_detection():
    """Stop detection"""
    global detection_active
    detection_active = False
    stop_capture()
    return jsonify({'status': 'stopped'})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Upload video file for detection"""
    global video_mode, current_video_path, detection_active, ip_mode
    
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
        ip_mode = False
        current_video_path = filepath
        restart_capture()
        
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
    global video_mode, current_video_path, detection_active, ip_mode
    
    # Stop current detection
    detection_active = False
    time.sleep(0.5)
    
    # Switch to camera
    video_mode = False
    ip_mode = False
    current_video_path = None
    detection_active = True
    restart_capture()
    
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
        'current_video': os.path.basename(current_video_path) if current_video_path else None,
        'ip_mode': ip_mode,
        'ip_camera_url': ip_camera_url,
        'camera_index': camera_index,
        'source': stats.get('source'),
        'single_save_mode': settings.get('single_save_mode', True)
    })

@app.route('/reset_saves', methods=['POST'])
def reset_saves():
    """Reset session memory for single-save mode (objects and faces)."""
    saved_object_labels.clear()
    saved_face_ids.clear()
    return jsonify({'status': 'reset', 'objects': 0, 'faces': 0})

@app.route('/set_save_mode', methods=['POST'])
def set_save_mode():
    """Enable or disable saving based on current page."""
    global save_enabled
    data = request.get_json()
    save_enabled = data.get('enabled', False)
    return jsonify({'status': 'success', 'save_enabled': save_enabled})

@app.route('/list_cameras')
def list_cameras():
    """Probe camera indices and return which ones open."""
    found = []
    for idx in range(0, 7):  # probe 0-6
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        if ok:
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            found.append({'index': idx, 'width': int(w), 'height': int(h)})
        cap.release()
    return jsonify({'cameras': found})

@app.route('/test_camera_index', methods=['POST'])
def test_camera_index():
    data = request.json or {}
    try:
        idx = int(data.get('index', 0))
    except Exception:
        return jsonify({'status': 'error', 'message': 'Invalid index'}), 400
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            return jsonify({'status': 'success', 'index': idx})
        return jsonify({'status': 'error', 'message': 'Opened but no frame', 'index': idx}), 500
    else:
        cap.release()
        return jsonify({'status': 'error', 'message': 'Cannot open', 'index': idx}), 404

@app.route('/camera_index', methods=['POST'])
def set_camera_index():
    """Set the active camera index and switch to camera source."""
    global camera_index, video_mode, ip_mode, detection_active
    try:
        data = request.json or {}
        idx = int(data.get('index', 0))
        camera_index = max(0, idx)
        video_mode = False
        ip_mode = False
        detection_active = True
        restart_capture()
        return jsonify({'status': 'success', 'camera_index': camera_index})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/use_ip_camera', methods=['POST'])
def use_ip_camera():
    """Use an IP camera URL (MJPEG/RTSP) as source."""
    global ip_mode, ip_camera_url, video_mode, detection_active
    data = request.json or {}
    url = data.get('url', '').strip()
    if not url or not (url.startswith('http://') or url.startswith('https://') or url.startswith('rtsp://')):
        return jsonify({'status': 'error', 'message': 'Invalid URL. Use http(s):// or rtsp://'}), 400
    ip_camera_url = url
    ip_mode = True
    video_mode = False
    detection_active = True
    restart_capture()
    return jsonify({'status': 'success', 'ip_camera_url': ip_camera_url})

@app.route('/gallery/<folder_type>')
def gallery(folder_type):
    """Get gallery images from objects or faces folder"""
    try:
        if folder_type == 'objects':
            base_dir = OBJECTS_DIR
        elif folder_type == 'faces':
            base_dir = FACES_DIR
        else:
            return jsonify({'status': 'error', 'message': 'Invalid folder type'}), 400
        
        images = []
        categories = {}
        
        if os.path.exists(base_dir):
            # Get all subdirectories
            for subdir in os.listdir(base_dir):
                subdir_path = os.path.join(base_dir, subdir)
                if os.path.isdir(subdir_path):
                    category_count = 0
                    # Get all images in subdirectory
                    for filename in os.listdir(subdir_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            file_path = os.path.join(subdir_path, filename)
                            # Get file modification time
                            try:
                                mod_time = os.path.getmtime(file_path)
                                date_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                                
                                # Create relative path for web access
                                relative_path = f"/datasets/{folder_type}/{subdir}/{filename}"
                                
                                images.append({
                                    'name': subdir,
                                    'filename': filename,
                                    'path': relative_path,
                                    'date': date_str,
                                    'timestamp': mod_time
                                })
                                category_count += 1
                            except Exception as e:
                                print(f"Error processing {file_path}: {e}")
                    
                    if category_count > 0:
                        categories[subdir] = category_count
        
        # Sort by timestamp (newest first)
        images.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Limit to most recent 500 images for faster loading
        total_count = len(images)
        images = images[:500]
        
        print(f"📊 Gallery {folder_type}: Showing {len(images)} of {total_count} images in {len(categories)} categories")
        print(f"📁 Top categories: {dict(list(categories.items())[:5])}")
        
        return jsonify({
            'status': 'success', 
            'images': images, 
            'count': len(images),
            'total_count': total_count,
            'categories': categories
        })
    except Exception as e:
        print(f"❌ Gallery error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/gallery/latest')
def gallery_latest():
    """Return lightweight summary of gallery counts and latest timestamps for live polling."""
    try:
        obj_count = 0
        face_count = 0
        obj_latest = 0
        face_latest = 0

        # Objects
        if os.path.exists(OBJECTS_DIR):
            for subdir in os.listdir(OBJECTS_DIR):
                subdir_path = os.path.join(OBJECTS_DIR, subdir)
                if os.path.isdir(subdir_path):
                    for filename in os.listdir(subdir_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            obj_count += 1
                            try:
                                ts = os.path.getmtime(os.path.join(subdir_path, filename))
                                if ts > obj_latest:
                                    obj_latest = ts
                            except Exception:
                                pass

        # Faces
        if os.path.exists(FACES_DIR):
            for subdir in os.listdir(FACES_DIR):
                subdir_path = os.path.join(FACES_DIR, subdir)
                if os.path.isdir(subdir_path):
                    for filename in os.listdir(subdir_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            face_count += 1
                            try:
                                ts = os.path.getmtime(os.path.join(subdir_path, filename))
                                if ts > face_latest:
                                    face_latest = ts
                            except Exception:
                                pass

        return jsonify({
            'status': 'success',
            'objects': obj_count,
            'faces': face_count,
            'objects_latest': obj_latest,
            'faces_latest': face_latest
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/datasets/<folder_type>/<subfolder>/<filename>')
def serve_dataset_image(folder_type, subfolder, filename):
    """Serve images from datasets folder"""
    try:
        if folder_type == 'objects':
            base_dir = OBJECTS_DIR
        elif folder_type == 'faces':
            base_dir = FACES_DIR
        else:
            return "Invalid folder type", 404
        
        file_path = os.path.join(base_dir, subfolder, filename)
        
        if os.path.exists(file_path):
            from flask import send_file
            return send_file(file_path, mimetype='image/jpeg')
        else:
            return "Image not found", 404
    except Exception as e:
        return str(e), 500

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
    
    # Use debug=False to avoid multi-process auto-reloader interfering with camera
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
