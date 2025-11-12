# 🚀 Quick Start Guide

## DTP AI Project - Object Detector
**SMK Telkom Sidoarjo - DTP AI Specialist**

---

## 📦 Apa isi folder ini?

Ini adalah **project utama** untuk Object Detection dengan fitur:
- ✅ Real-time object detection (YOLO)
- ✅ Face detection
- ✅ Region of Interest (ROI) analysis
- ✅ Upload & process video files
- ✅ Web interface yang modern dan elegan

---

## 🏃 Cara Menjalankan (CEPAT!)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Jalankan Server
```bash
python app.py
```

### 3️⃣ Buka Browser
Akses: **http://localhost:5000**

---

## 📁 Struktur Folder

```
DTP_AI_Object_Detector/
├── app.py                    # ⭐ Main Flask application
├── requirements.txt          # 📦 Dependencies
├── README.md                 # 📖 Dokumentasi lengkap
├── START_HERE.md            # 🚀 File ini - Quick start
├── ROI_YOLO_Detection.ipynb # 📓 Jupyter notebook version
│
├── templates/               # 🎨 HTML templates
│   └── index.html          # Main web interface
│
├── static/                  # 📂 Static files (future)
│   ├── css/                # CSS files
│   ├── js/                 # JavaScript files
│   └── images/             # Images
│
├── models/                  # 🤖 AI Models
│   └── yolov8n.pt          # YOLO model
│
└── uploads/                 # 📹 Uploaded videos
    └── (video files...)
```

---

## 🎯 Fitur Utama

### 1. Object Detection
- Deteksi 80+ objek (person, book, bottle, phone, car, dll)
- YOLO v8 nano model (cepat & akurat)
- Real-time dari webcam atau video file

### 2. Face Detection
- Deteksi wajah dengan kotak magenta
- Haar Cascade classifier
- Real-time tracking

### 3. ROI (Region of Interest)
- 5 posisi preset: Center, Top, Bottom, Left, Right
- Hitung objek dalam/luar ROI
- Visual feedback dengan color-coding

### 4. Video Processing
- Upload video (MP4, AVI, MOV, dll)
- Max 100MB per file
- Auto-loop playback

### 5. Web Interface
- Purple gradient design
- Glass-morphism effects
- Real-time statistics
- Interactive controls

---

## ⚙️ System Requirements

### Minimum:
- Python 3.8+
- 4GB RAM
- Webcam (untuk real-time detection)
- Browser modern (Chrome/Firefox/Edge)

### Recommended:
- Python 3.10 atau 3.11
- 8GB RAM
- GPU (CUDA) untuk processing lebih cepat
- Good lighting untuk detection optimal

---

## 🔧 Troubleshooting

### ❌ "localhost refused to connect"
**Solusi:** Server belum jalan. Jalankan `python app.py` dulu

### ❌ "ModuleNotFoundError"
**Solusi:** Install dependencies dengan `pip install -r requirements.txt`

### ❌ Webcam tidak terdeteksi
**Solusi:** 
- Tutup aplikasi lain yang pakai webcam
- Allow camera permission di browser
- Restart aplikasi

### ❌ FPS rendah / lag
**Solusi:**
- Tutup aplikasi lain
- Turunkan confidence threshold
- Disable face detection jika tidak perlu

---

## 📚 Dokumentasi Lengkap

Baca **README.md** untuk:
- Instalasi detail
- Penjelasan semua fitur
- Advanced configuration
- API documentation
- Development guide

---

## 👨‍💻 Development

### Modify UI
Edit: `templates/index.html`

### Modify Backend
Edit: `app.py`

### Add Static Files
Taruh di: `static/css/`, `static/js/`, `static/images/`

### Add New Model
Taruh di: `models/` folder

---

## 🎓 Credits

**Created by:** SMK Telkom Sidoarjo - DTP AI Specialist  
**Project:** Object Detection with Face Recognition  
**Framework:** Flask + YOLO + OpenCV  
**Year:** 2025

---

## 📞 Support

Jika ada masalah:
1. Cek **README.md** untuk troubleshooting lengkap
2. Lihat error message di terminal
3. Pastikan semua dependencies terinstall
4. Cek webcam dan browser permission

---

## 🎉 Selamat Mencoba!

**Happy Detecting! 🚀**

---

*File ini dibuat untuk memudahkan quick start project*  
*Untuk dokumentasi lengkap, baca README.md*
