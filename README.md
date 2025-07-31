"# 👥 Face Recognition Attendance System

A modern, fast, and comprehensive face recognition-based attendance system with a unified Streamlit web interface. Features optimized sampling, photo upload support, and real-time attendance tracking.

## 🚀 Key Features

- ✅ **Fast Registration**: Only 20 samples needed (vs 100+ in traditional systems)
- ✅ **Photo Upload**: Register people instantly by uploading 3-10 photos
- ✅ **Smart Data Augmentation**: Automatically generates variations for better accuracy
- ✅ **Unified Web Interface**: All features in one Streamlit application
- ✅ **Real-time Recognition**: Live camera-based attendance marking
- ✅ **Interactive Dashboard**: View attendance statistics and charts
- ✅ **Export Capabilities**: Download attendance data and registration summaries
- ✅ **Duplicate Prevention**: Smart system prevents double attendance marking

## ⚡ Performance Improvements

| Feature | Old System | New System | Improvement |
|---------|------------|------------|-------------|
| Sample Collection | 100 samples (~5+ minutes) | 20 samples (~30-60 seconds) | **90% faster** |
| Registration Method | Camera only | Camera + Photo Upload | **Multiple options** |
| Data Augmentation | None | Automatic variations | **Better accuracy** |
| Interface | Separate files | Unified web app | **Streamlined** |

## 📁 Project Structure

```
face-recognition-attendance-system/
├── streamlit_app.py          # Main unified application
├── requirements.txt          # Python dependencies
├── data/
│   ├── face.xml             # Haar cascade for face detection
│   ├── names.pkl            # Stored names
│   └── faces_data.pkl       # Stored face encodings
├── Attendence/
│   └── Attendance_YYYY-MM-DD.csv  # Daily attendance records
└── README.md                # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.7+
- Webcam/Camera (for live capture)
- Windows/Linux/Mac OS

### Quick Setup
1. **Clone the repository**
```bash
git clone <repository-url>
cd face-recognition-attendance-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## 📋 Usage Guide

### 🎯 Navigation
The application has 4 main pages accessible via the sidebar:

1. **📊 Dashboard** - View attendance records and statistics
2. **👤 Add Faces** - Register new people (Camera or Photo Upload)
3. **📹 Mark Attendance** - Real-time attendance marking
4. **⚙️ Settings** - System configuration and data management

### 1️⃣ Adding People (2 Methods)

#### Method A: Fast Camera Capture (20 samples)
1. Go to "👤 Add Faces" page
2. Enter person's name
3. Click "📹 Start Camera"
4. Select "📹 Live Camera (20 samples)"
5. Position face in camera view
6. Wait ~30-60 seconds for automatic capture
7. System generates additional variations automatically

#### Method B: Photo Upload (Fastest)
1. Go to "👤 Add Faces" page
2. Enter person's name
3. Click "📹 Start Camera"
4. Select "📤 Upload Photos"
5. Upload 3-10 clear photos of the person
6. Click "🔄 Process Photos"
7. System processes photos in seconds

### 2️⃣ Marking Attendance

#### Automatic Recognition
1. Go to "📹 Mark Attendance" page
2. Click "📹 Start Recognition"
3. Face appears in camera → Automatic attendance marking
4. System prevents duplicate entries

#### Manual Attendance
1. Go to "📹 Mark Attendance" page
2. Select person from dropdown
3. Click "✅ Mark Present"

### 3️⃣ Viewing Reports
1. Go to "📊 Dashboard" page
2. Select date to view attendance
3. View statistics, charts, and detailed records
4. Export data if needed

## 🔧 Advanced Features

### Smart Data Augmentation
The system automatically creates variations from each sample:
- **Rotations**: ±3°, ±5° for different head positions
- **Brightness**: 80%, 90%, 110%, 120% variations
- **Original**: Plus the original sample
- **Result**: 9 variations per original sample

### Export Options
- **Attendance Archive**: ZIP file with all CSV files
- **Registration Summary**: CSV with people and sample counts
- **Date-specific Data**: Individual daily reports

### System Management
- **Clear Face Data**: Remove all registered people
- **Reset Attendance**: Clear today's attendance records
- **Refresh Data**: Reload all caches and models

## 📊 Technical Specifications

| Component | Technology | Details |
|-----------|------------|---------|
| **Face Detection** | OpenCV Haar Cascades | Real-time face detection |
| **Face Recognition** | K-Nearest Neighbors (KNN) | 5-neighbor classifier |
| **Image Processing** | 50x50 grayscale | Consistent 2500-feature vectors |
| **Web Framework** | Streamlit | Interactive web interface |
| **Data Storage** | Pickle + CSV | Face data and attendance records |
| **Data Augmentation** | OpenCV transformations | Rotations and brightness variations |

## 🚨 Troubleshooting

### Common Issues & Solutions

1. **Camera not working**
   ```
   ❌ Could not read from camera
   ```
   - Check if camera is connected
   - Close other applications using camera
   - Try restarting the application

2. **Face not detected in photos**
   ```
   ⚠️ No face detected in filename.jpg
   ```
   - Use clear, well-lit photos
   - Ensure face is clearly visible
   - Try photos with different angles

3. **Need more samples error**
   ```
   ❌ Need at least 15 samples. Only got X.
   ```
   - Upload more photos (3-10 recommended)
   - Use camera capture as backup
   - Ensure photos contain clear faces

4. **Face detector not found**
   ```
   ❌ face.xml not found! Please run setup first.
   ```
   - Download `haarcascade_frontalface_default.xml` from OpenCV
   - Rename to `face.xml`
   - Place in `data/` folder

### Performance Tips
- **Good Lighting**: Ensure adequate lighting for face detection
- **Clear Photos**: Use high-quality, clear photos for upload
- **Variety**: Include different angles and expressions
- **Regular Cleanup**: Periodically clear old attendance data

## 📦 Dependencies

```txt
streamlit>=1.47.1
opencv-python>=4.12.0
scikit-learn>=1.7.1
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
```

## 🔄 Migration from Old System

If you have data from previous versions:
1. Place existing `names.pkl` and `faces_data.pkl` in `data/` folder
2. Move CSV files to `Attendence/` folder
3. Download `face.xml` to `data/` folder
4. Run the new unified application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Test thoroughly with the unified interface
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/new-feature`)
7. Create a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the Settings page in the application
3. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System information (OS, Python version)

---

**Made with ❤️ using Streamlit, OpenCV, and scikit-learn**" 
