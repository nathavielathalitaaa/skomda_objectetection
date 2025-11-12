# 🧠 DTP AI Project - Object Detector

**Created by: SMK Telkom Sidoarjo - DTP AI Specialist**

Aplikasi web interaktif untuk deteksi objek real-time menggunakan YOLOv8 dengan Region of Interest (ROI) dan Face Emotion Recognition menggunakan DeepFace.

---

## ✨ Fitur Utama

### 🎯 Object Detection (YOLO)
- **Real-time Detection**: Deteksi objek dari webcam atau video file
- **ROI Analysis**: Fokus deteksi pada area tertentu (Center, Top, Bottom, Left, Right)
- **Multi-object Detection**: Deteksi berbagai objek seperti:
  - 👤 Person (Orang)
  - 📚 Book (Buku)
  - 👓 Glasses (Kacamata)  
  - 🍾 Bottle (Botol)
  - ⌚ Watch (Jam tangan/Gelang)
  - 📱 Cell phone (HP)
  - 💼 Bag (Tas)
  - 🎒 Backpack (Ransel)
  - ☂️ Umbrella (Payung)
  - 👔 Tie (Dasi)
  - Dan 70+ objek lainnya dari COCO dataset

### 😊 Face Emotion Recognition
- **Deteksi Wajah**: Menggunakan Haar Cascade Classifier
- **Analisis Emosi**: Mendeteksi 7 ekspresi wajah:
  - 😠 **Marah** (Angry)
  - 🤢 **Jijik** (Disgust)
  - 😨 **Takut** (Fear)
  - 😄 **Senang** (Happy)
  - 😢 **Sedih** (Sad)
  - 😲 **Terkejut** (Surprise)
  - 😐 **Netral** (Neutral)
- **Real-time Processing**: Analisis emosi dalam waktu nyata
- **Visual Feedback**: Kotak magenta untuk deteksi wajah dengan label emosi

### 🎥 Video Source Options
- **Webcam**: Deteksi real-time dari kamera laptop/PC
- **Upload Video**: Support berbagai format (MP4, AVI, MOV, MKV, WMV, FLV, WEBM)
- **Easy Switching**: Beralih antara webcam dan video dengan mudah
- **Auto Loop**: Video otomatis diulang saat selesai

### ⚙️ Pengaturan Fleksibel
- **ROI Position**: 5 posisi preset (Center, Top, Bottom, Left, Right)
- **Confidence Threshold**: Atur sensitivitas deteksi (0.0 - 1.0)
- **Toggle Options**:
  - Show/Hide ROI Box
  - Show/Hide Object Labels
  - Enable/Disable Face Emotion Detection
- **Real-time Updates**: Semua pengaturan dapat diubah saat deteksi berjalan

### 📊 Statistik Real-time
- **Objects in ROI**: Jumlah objek dalam area ROI
- **Objects Outside ROI**: Jumlah objek di luar ROI
- **Total Detections**: Total objek terdeteksi
- **FPS**: Frames per second
- **Detected Objects**: List objek dengan jumlahnya
- **Detected Emotions**: List emosi wajah yang terdeteksi

### 🎨 UI Modern & Elegan
- **Gradient Design**: Purple gradient background yang menarik
- **Glass-morphism**: Efek kaca transparan pada UI elements
- **Responsive Layout**: Tampilan optimal di berbagai ukuran layar
- **Smooth Animations**: Transisi dan animasi yang halus
- **Font Awesome Icons**: Icon-icon yang intuitif
- **Color-coded Detection**:
  - 🔴 Red Box = ROI Area
  - 🟢 Green Box = Objects in ROI
  - 🟠 Orange Box = Objects Outside ROI
  - 🟣 Magenta Box = Face Detection

---

## 🚀 Instalasi

### Prerequisites
- Python 3.8 atau lebih tinggi
- Webcam (untuk mode real-time)
- Browser modern (Chrome, Firefox, Edge)

### 1. Clone atau Download Project
```bash
cd DTP_AI
```

### 2. Buat Virtual Environment (Recommended)
```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Catatan**: Instalasi pertama kali akan download beberapa model:
- YOLOv8 model (~6 MB)
- DeepFace emotion model (~100 MB)
- TensorFlow (~500 MB)

### 4. Jalankan Aplikasi
```bash
python app.py
```

### 5. Akses di Browser
Buka browser dan ketik: `http://localhost:5000`

---

## 📝 Cara Menggunakan

### Mode 1: Webcam (Default)
1. Pastikan webcam terhubung dan diizinkan oleh browser
2. Akses aplikasi di `http://localhost:5000`
3. Video stream akan otomatis dimulai
4. Wajah akan terdeteksi dengan kotak magenta dan label emosi
5. Objek akan terdeteksi dengan kotak hijau (dalam ROI) atau orange (luar ROI)

### Mode 2: Upload Video
1. Klik tombol **"Upload Video File"**
2. Pilih file video dari komputer (max 100MB)
3. Tunggu proses upload selesai
4. Video akan otomatis diproses dengan deteksi objek dan emosi
5. Video akan loop otomatis

### Mode 3: Kembali ke Webcam
1. Klik tombol **"Switch to Camera"**
2. Konfirmasi dialog
3. Aplikasi kembali ke mode webcam

### Pengaturan ROI
1. **Pilih ROI Position**:
   - Center: ROI di tengah frame (recommended)
   - Top: ROI di bagian atas
   - Bottom: ROI di bagian bawah
   - Left: ROI di sisi kiri
   - Right: ROI di sisi kanan

2. **Sesuaikan Confidence Threshold**:
   - 0.3-0.4: Deteksi lebih banyak (mungkin ada false positive)
   - 0.5-0.6: Balance (recommended)
   - 0.7-0.9: Hanya objek dengan confidence tinggi

3. **Toggle Options**:
   - ✅ Show ROI Box: Tampilkan kotak ROI merah
   - ✅ Show Labels: Tampilkan label objek
   - ✅ Detect Face Emotion: Aktifkan deteksi emosi wajah

4. **Apply Settings**: Klik tombol untuk menerapkan perubahan

### Kontrol Aplikasi
- **Upload Video File**: Upload video untuk analisis
- **Switch to Camera**: Beralih ke mode webcam
- **Stop Camera/Video**: Hentikan deteksi sementara
- **Apply Settings**: Terapkan pengaturan baru

---

## 📁 Struktur Project

```
DTP_AI/
├── app.py                          # Flask backend + AI logic
├── templates/
│   └── index.html                  # Frontend UI
├── uploads/                        # Folder video uploads (auto-created)
├── yolov8n.pt                      # YOLO model (auto-download)
├── requirements.txt                # Python dependencies
├── README_UPDATED.md               # Dokumentasi lengkap
└── .venv/                          # Virtual environment (optional)
```

---

## 🎯 Objek yang Dapat Dideteksi

YOLOv8 dapat mendeteksi 80 kelas objek dari COCO dataset, termasuk:

### 👤 Manusia & Aksesori
- Person, Eye glasses, Sun glasses, Tie, Umbrella, Handbag, Suitcase, Backpack

### 🚗 Kendaraan
- Bicycle, Car, Motorcycle, Airplane, Bus, Train, Truck, Boat

### 🐾 Hewan
- Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe

### 🍔 Makanan & Minuman
- Bottle, Wine glass, Cup, Fork, Knife, Spoon, Bowl, Banana, Apple, Sandwich, Orange, Pizza, Donut, Cake

### 💼 Peralatan Kantor & Rumah
- Book, Clock, Vase, Scissors, Teddy bear, Hair dryer, Toothbrush, Laptop, Mouse, Keyboard, Cell phone, TV, Remote

### ⚽ Olahraga
- Sports ball, Baseball bat, Tennis racket, Frisbee, Skateboard, Surfboard, Ski

### 🪑 Furniture
- Chair, Couch, Bed, Dining table, Toilet, Potted plant

---

## 😊 Ekspresi Wajah yang Dideteksi

DeepFace menggunakan model deep learning untuk mengidentifikasi 7 ekspresi dasar:

| Ekspresi | Deskripsi | Emoji |
|----------|-----------|-------|
| **Marah** (Angry) | Wajah menunjukkan kemarahan | 😠 |
| **Jijik** (Disgust) | Ekspresi jijik/tidak suka | 🤢 |
| **Takut** (Fear) | Wajah menunjukkan ketakutan | 😨 |
| **Senang** (Happy) | Senyum/ekspresi gembira | 😄 |
| **Sedih** (Sad) | Wajah murung/sedih | 😢 |
| **Terkejut** (Surprise) | Ekspresi terkejut | 😲 |
| **Netral** (Neutral) | Wajah tanpa ekspresi khusus | 😐 |

---

## 🔧 Troubleshooting

### ❌ Webcam tidak terdeteksi
**Solusi**:
1. Pastikan webcam terhubung dengan benar
2. Tutup aplikasi lain yang menggunakan webcam (Zoom, Teams, dll)
3. Periksa permission webcam di browser (klik ikon gembok di address bar)
4. Restart browser atau aplikasi Flask
5. Coba browser lain

### ❌ Video tidak bisa diupload
**Solusi**:
1. Periksa format video (harus: mp4, avi, mov, mkv, wmv, flv, webm)
2. Pastikan ukuran file tidak melebihi 100MB
3. Compress video jika terlalu besar
4. Periksa koneksi internet

### ❌ Emotion detection tidak jalan
**Solusi**:
1. Pastikan checkbox "Detect Face Emotion" di-centang
2. Klik "Apply Settings"
3. Pastikan wajah terlihat jelas di kamera (pencahayaan cukup)
4. Tunggu model DeepFace download (pertama kali)
5. Check terminal untuk error message

### ❌ FPS rendah / lag
**Solusi**:
1. Tutup aplikasi lain yang berat
2. Matikan emotion detection jika tidak diperlukan
3. Tingkatkan confidence threshold (0.6-0.8)
4. Gunakan model YOLOv8n (nano) - sudah default
5. Kurangi resolusi webcam di `app.py`

### ❌ Error saat install dependencies
**Solusi**:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install wheel
pip install wheel

# Install ulang requirements
pip install -r requirements.txt --no-cache-dir
```

### ❌ Port 5000 sudah digunakan
**Solusi**:
Edit `app.py` baris terakhir:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Ganti ke port lain
```

### ❌ TensorFlow error pada Windows
**Solusi**:
Install Microsoft Visual C++ Redistributable:
https://aka.ms/vs/16/release/vc_redist.x64.exe

---

## 💡 Tips untuk Hasil Terbaik

### 📸 Untuk Object Detection:
1. **Pencahayaan**: Pastikan ruangan terang untuk hasil optimal
2. **Jarak Objek**: Tidak terlalu dekat/jauh dari kamera (50cm - 2m ideal)
3. **Background**: Latar belakang sederhana lebih mudah dideteksi
4. **Confidence**: Mulai dengan 0.5, sesuaikan berdasarkan hasil

### 😊 Untuk Emotion Detection:
1. **Pencahayaan Wajah**: Cahaya merata di wajah (tidak backlight)
2. **Posisi Wajah**: Menghadap kamera (tidak menyamping)
3. **Ukuran Wajah**: Wajah cukup besar di frame (tidak terlalu jauh)
4. **Ekspresi Jelas**: Buat ekspresi yang jelas untuk akurasi tinggi
5. **Satu Wajah**: Hasil terbaik dengan satu wajah per frame

### 🎯 Untuk ROI:
1. **Pilih Area Strategis**: Sesuaikan ROI dengan area monitoring
2. **Center ROI**: Cocok untuk monitoring area di depan kamera
3. **Custom ROI**: Untuk kebutuhan spesifik (coming soon)

---

## 📊 Performance

### Kecepatan (FPS):
- **Webcam Only**: ~25-30 FPS
- **With Object Detection**: ~15-20 FPS
- **With Emotion Detection**: ~10-15 FPS
- **Both Active**: ~8-12 FPS

**Note**: FPS tergantung spesifikasi komputer

### Resource Usage:
- **CPU**: 30-50% (dengan TensorFlow CPU)
- **RAM**: ~2-3 GB
- **GPU** (jika ada): ~1-2 GB VRAM

---

## 🛠️ Technologies Used

### Backend:
- **Flask**: Web framework
- **OpenCV**: Computer vision & video processing
- **Ultralytics YOLOv8**: Object detection
- **DeepFace**: Face emotion recognition
- **TensorFlow**: Deep learning backend
- **NumPy**: Numerical operations

### Frontend:
- **HTML5**: Structure
- **CSS3**: Styling (Gradient, Glass-morphism)
- **JavaScript**: Interactive features
- **Font Awesome**: Icons
- **Google Fonts**: Poppins typography

### AI Models:
- **YOLOv8n**: Nano model untuk kecepatan (6MB)
- **DeepFace**: Emotion detection model (~100MB)
- **Haar Cascade**: Face detection

---

## 📜 Lisensi & Credits

### Licenses:
- YOLOv8: AGPL-3.0 (Ultralytics)
- DeepFace: MIT License
- TensorFlow: Apache 2.0

### Credits:
- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **DeepFace**: [serengil](https://github.com/serengil/deepface)
- **OpenCV**: [OpenCV.org](https://opencv.org/)

---

## 👨‍💻 Pengembang

**DTP AI Specialist Team**  
SMK Telkom Sidoarjo  
Departemen Teknik Perangkat AI

**Project Type**: Educational & Learning  
**Purpose**: Object Detection with Emotion Recognition Demo

---

## 🔮 Fitur Mendatang (Future Updates)

- [ ] Custom ROI dengan drag-and-drop
- [ ] Export hasil deteksi ke CSV/JSON
- [ ] Download video hasil deteksi
- [ ] Multiple ROI zones (multi-area)
- [ ] Object tracking dengan ID unik
- [ ] Notifikasi saat objek/emosi tertentu terdeteksi
- [ ] Historical data & analytics dashboard
- [ ] Age & Gender detection
- [ ] Mask detection
- [ ] Social distancing monitoring
- [ ] Custom YOLO model training interface
- [ ] Multi-camera support
- [ ] Cloud storage integration

---

## 📞 Support & Contact

Untuk pertanyaan, bug report, atau feature request:
- **Email**: dtp.ai@smktelkom-sby.sch.id (example)
- **GitHub Issues**: Create new issue di repository
- **Documentation**: Baca README dan kode comments

---

## 🙏 Acknowledgments

Terima kasih kepada:
- SMK Telkom Sidoarjo atas dukungan fasilitas
- Tim pengajar DTP AI Specialist
- Komunitas open-source: Ultralytics, DeepFace, OpenCV
- Semua kontributor dan pengguna aplikasi ini

---

**🎉 Happy Detecting! Selamat Mencoba!**

---

**© 2025 SMK Telkom Sidoarjo - DTP AI Specialist**  
*Building the Future with Artificial Intelligence*
