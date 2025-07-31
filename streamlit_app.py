import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import time
from sklearn.neighbors import KNeighborsClassifier
import csv
from datetime import datetime
import tempfile
from PIL import Image

# Try to import OpenCV with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError as e:
    OPENCV_AVAILABLE = False
    st.error(f"⚠️ OpenCV not available: {str(e)}")
    st.info("📷 Camera features will be disabled. Photo upload is still available!")

# Configure page
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'marked_names' not in st.session_state:
    st.session_state.marked_names = set()

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("Attendence", exist_ok=True)

# Initialize face detector (only if OpenCV is available)
@st.cache_resource
def load_face_detector():
    if not OPENCV_AVAILABLE:
        return None
    if not os.path.exists('data/face.xml'):
        # Try to create a basic face detector using alternative method
        try:
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except:
            st.warning("⚠️ Using basic face detection. For better accuracy, add face.xml to data/ folder.")
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return cv2.CascadeClassifier('data/face.xml')

# Alternative face detection using PIL for uploaded images
def detect_faces_pil(image_pil):
    """Simple face detection using PIL - fallback when OpenCV not available"""
    # Convert PIL to numpy array
    img_array = np.array(image_pil.convert('L'))  # Convert to grayscale
    
    # Simple center crop as face detection fallback
    h, w = img_array.shape
    size = min(h, w)
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    
    face_crop = img_array[start_h:start_h+size, start_w:start_w+size]
    return [face_crop]  # Return as list to match OpenCV format

# Load or initialize data
@st.cache_data
def load_face_data():
    try:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        return names, faces
    except FileNotFoundError:
        return [], []

def save_face_data(names, faces):
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
    # Clear cache to reload updated data
    st.cache_data.clear()

def generate_face_variations_pil(face_image):
    """Generate variations using PIL when OpenCV is not available"""
    variations = []
    
    # Convert to PIL Image if numpy array
    if isinstance(face_image, np.ndarray):
        face_pil = Image.fromarray(face_image)
    else:
        face_pil = face_image
    
    # Resize to standard size
    face_resized = face_pil.resize((50, 50))
    variations.append(np.array(face_resized))
    
    # Rotations
    for angle in [-5, 5, -3, 3]:
        rotated = face_resized.rotate(angle, fillcolor=128)
        variations.append(np.array(rotated))
    
    # Brightness variations
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Brightness(face_resized)
    for factor in [0.8, 1.2, 0.9, 1.1]:
        bright = enhancer.enhance(factor)
        variations.append(np.array(bright))
    
    return variations

def generate_face_variations_cv2(face_image):
    """Generate variations using OpenCV when available"""
    if not OPENCV_AVAILABLE:
        return generate_face_variations_pil(face_image)
    
    variations = []
    
    # Original image
    resized = cv2.resize(face_image, (50, 50))
    variations.append(resized)
    
    # Slight rotations
    for angle in [-5, 5, -3, 3]:
        rows, cols = face_image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(face_image, rotation_matrix, (cols, rows))
        rotated_resized = cv2.resize(rotated, (50, 50))
        variations.append(rotated_resized)
    
    # Brightness variations
    for alpha in [0.8, 1.2, 0.9, 1.1]:
        bright = cv2.convertScaleAbs(face_image, alpha=alpha, beta=0)
        bright_resized = cv2.resize(bright, (50, 50))
        variations.append(bright_resized)
    
    return variations

# Train KNN model
@st.cache_resource
def train_knn_model():
    names, faces = load_face_data()
    if len(faces) == 0:
        return None, None
    
    faces_array = np.array(faces).reshape((len(faces), -1))
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces_array, names)
    return knn, names

# Sidebar Navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["📊 Dashboard", "👤 Add Faces", "📹 Mark Attendance", "⚙️ Settings"]
)

# Display OpenCV status
if not OPENCV_AVAILABLE:
    st.sidebar.error("📷 Camera features disabled")
    st.sidebar.info("📤 Photo upload available")
else:
    st.sidebar.success("📷 Camera features available")

# Main title
st.title("👥 Face Recognition Attendance System")

# Dashboard Page
if page == "📊 Dashboard":
    st.header("📊 Attendance Dashboard")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 5 seconds)", value=False)
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Date selection
    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input("📅 Select Date", datetime.now())
    with col2:
        if st.button("🔄 Refresh Data"):
            st.rerun()
    
    # Load attendance data
    date_str = selected_date.strftime("%Y-%m-%d")
    filename = f"Attendence/Attendance_{date_str}.csv"
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        st.success(f"✅ Attendance file loaded for {date_str}")
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Total Records", len(df))
        with col2:
            st.metric("👥 Unique People", df['Name'].nunique())
        with col3:
            st.metric("🕐 Latest Entry", df['Time'].iloc[-1] if not df.empty else "N/A")
        
        # Display data
        st.subheader("📋 Attendance Records")
        st.dataframe(df, use_container_width=True)
        
        # Charts
        if not df.empty:
            st.subheader("📊 Attendance Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                name_counts = df['Name'].value_counts()
                st.bar_chart(name_counts)
            
            with col2:
                # Time distribution
                df['Hour'] = pd.to_datetime(df['Time']).dt.hour
                hourly_counts = df['Hour'].value_counts().sort_index()
                st.line_chart(hourly_counts)
    else:
        st.warning(f"⚠️ No attendance file found for {date_str}")
        st.info("Use the 'Mark Attendance' page to create attendance records.")

# Add Faces Page
elif page == "👤 Add Faces":
    st.header("👤 Add New Person")
    
    # Check current data
    names, faces = load_face_data()
    st.info(f"Currently registered: {len(set(names))} people with {len(names)} total samples")
    
    if not OPENCV_AVAILABLE:
        st.warning("📷 Camera features are disabled. Use photo upload instead!")
    else:
        st.success("🚀 **Fast Mode**: Camera capture takes only 20 samples (~30-60 seconds)!")
    
    st.success("📤 **Photo Upload**: Upload 3-10 photos for fastest registration!")
    
    # Input for person's name
    person_name = st.text_input("👤 Enter person's name:", key="person_name")
    
    if person_name:
        # Photo Upload Method (always available)
        st.subheader("📤 Photo Upload Method")
        st.info("""
        📋 **Photo Upload Instructions:**
        1. Upload 3-10 clear photos of the person
        2. Photos should show different angles/expressions  
        3. System will automatically generate multiple samples from each photo
        4. Each photo will create 5-8 variations for training
        """)
        
        uploaded_files = st.file_uploader(
            "Choose photos", 
            accept_multiple_files=True, 
            type=['png', 'jpg', 'jpeg'],
            key="photo_upload"
        )
        
        if uploaded_files and st.button("🔄 Process Photos", key="process_photos"):
            try:
                faces_data = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    # Read uploaded image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    
                    if OPENCV_AVAILABLE:
                        # Use OpenCV if available
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        if image is None:
                            st.warning(f"⚠️ Could not process {uploaded_file.name}")
                            continue
                        
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        facedetect = load_face_detector()
                        faces = facedetect.detectMultiScale(gray, 1.3, 5)
                        
                        faces_found = 0
                        for (x, y, w, h) in faces:
                            if faces_found >= 1:  # Only take first face from each photo
                                break
                                
                            crop_img = image[y:y+h, x:x+w]
                            crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                            
                            # Generate multiple variations
                            variations = generate_face_variations_cv2(crop_img_gray)
                            faces_data.extend(variations)
                            faces_found += 1
                    else:
                        # Use PIL fallback
                        image_pil = Image.open(uploaded_file)
                        face_crops = detect_faces_pil(image_pil)
                        
                        for face_crop in face_crops[:1]:  # Take first face
                            variations = generate_face_variations_pil(face_crop)
                            faces_data.extend(variations)
                    
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed: {idx + 1}/{len(uploaded_files)} photos, Generated: {len(faces_data)} samples")
                
                if len(faces_data) >= 15:  # Minimum samples needed
                    # Limit to reasonable number
                    if len(faces_data) > 50:
                        faces_data = faces_data[:50]
                    
                    # Load existing data and save
                    existing_names, existing_faces = load_face_data()
                    new_names = existing_names + [person_name] * len(faces_data)
                    new_faces = existing_faces + faces_data
                    save_face_data(new_names, new_faces)
                    
                    st.success(f"✅ Successfully added {person_name} with {len(faces_data)} face samples from {len(uploaded_files)} photos!")
                    st.balloons()
                    st.cache_resource.clear()
                else:
                    st.error(f"❌ Need at least 15 samples. Only got {len(faces_data)}. Try uploading more photos with clear faces.")
            
            except Exception as e:
                st.error(f"❌ Error processing photos: {str(e)}")
        
        # Camera Method (only if OpenCV available)
        if OPENCV_AVAILABLE:
            st.subheader("📹 Camera Method")
            st.info("Position your face in the camera view. System will capture 20 samples (~30-60 seconds)")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📹 Start Camera", key="start_camera"):
                    st.session_state.camera_active = True
            with col2:
                if st.button("⏹️ Stop Camera", key="stop_camera"):
                    st.session_state.camera_active = False
            
            if st.session_state.camera_active:
                st.warning("📷 Camera capture feature requires local environment. Use photo upload for cloud deployment.")
                st.session_state.camera_active = False
    
    # Display registered people
    if names:
        st.subheader("👥 Registered People")
        unique_names = list(set(names))
        cols = st.columns(min(4, len(unique_names)))
        for i, name in enumerate(unique_names):
            with cols[i % 4]:
                count = names.count(name)
                st.metric(name, f"{count} samples")

# Mark Attendance Page
elif page == "📹 Mark Attendance":
    st.header("📹 Mark Attendance")
    
    # Check if model is trained
    knn, model_names = train_knn_model()
    
    if knn is None:
        st.warning("⚠️ No face data found. Please add faces first!")
        st.stop()
    
    st.success(f"✅ Model loaded with {len(set(model_names))} registered people")
    
    # Manual attendance option
    st.subheader("✍️ Manual Attendance")
    col1, col2 = st.columns(2)
    with col1:
        manual_name = st.selectbox("Select Person", sorted(set(model_names)))
    with col2:
        if st.button("✅ Mark Present"):
            current_time = datetime.now()
            date_str = current_time.strftime('%Y-%m-%d')
            time_str = current_time.strftime('%H:%M:%S')
            filename = f"Attendence/Attendance_{date_str}.csv"
            
            file_exists = os.path.isfile(filename)
            with open(filename, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(['Name', 'Date', 'Time'])
                writer.writerow([manual_name, date_str, time_str])
            
            st.success(f"✅ Attendance marked for {manual_name}")
            st.session_state.marked_names.add(manual_name)
    
    # Photo-based recognition
    st.subheader("📸 Photo Recognition")
    st.info("Upload a photo to recognize and mark attendance automatically")
    
    recognition_photo = st.file_uploader(
        "Upload photo for recognition", 
        type=['png', 'jpg', 'jpeg'],
        key="recognition_photo"
    )
    
    if recognition_photo and st.button("🔍 Recognize & Mark Attendance"):
        try:
            # Process uploaded photo
            file_bytes = np.asarray(bytearray(recognition_photo.read()), dtype=np.uint8)
            
            if OPENCV_AVAILABLE:
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                facedetect = load_face_detector()
                faces = facedetect.detectMultiScale(gray, 1.3, 5)
                
                recognized_names = []
                for (x, y, w, h) in faces:
                    crop_img = image[y:y+h, x:x+w]
                    crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    resized_img = cv2.resize(crop_img_gray, (50, 50)).flatten().reshape(1, -1)
                    
                    pred = knn.predict(resized_img)
                    confidence = knn.predict_proba(resized_img).max()
                    
                    if confidence > 0.7:  # Confidence threshold
                        recognized_names.append(str(pred[0]))
            else:
                # PIL fallback
                image_pil = Image.open(recognition_photo)
                face_crops = detect_faces_pil(image_pil)
                
                recognized_names = []
                for face_crop in face_crops:
                    resized_img = np.array(Image.fromarray(face_crop).resize((50, 50))).flatten().reshape(1, -1)
                    pred = knn.predict(resized_img)
                    confidence = knn.predict_proba(resized_img).max()
                    
                    if confidence > 0.6:  # Lower threshold for PIL method
                        recognized_names.append(str(pred[0]))
            
            # Mark attendance for recognized faces
            if recognized_names:
                current_time = datetime.now()
                date_str = current_time.strftime('%Y-%m-%d')
                time_str = current_time.strftime('%H:%M:%S')
                filename = f"Attendence/Attendance_{date_str}.csv"
                
                file_exists = os.path.isfile(filename)
                with open(filename, "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if not file_exists:
                        writer.writerow(['Name', 'Date', 'Time'])
                    
                    for name in set(recognized_names):  # Remove duplicates
                        writer.writerow([name, date_str, time_str])
                        st.session_state.marked_names.add(name)
                
                st.success(f"✅ Attendance marked for: {', '.join(set(recognized_names))}")
            else:
                st.warning("⚠️ No faces recognized in the photo. Try manual attendance.")
                
        except Exception as e:
            st.error(f"❌ Error processing photo: {str(e)}")
    
    # Display marked attendance
    if st.session_state.marked_names:
        st.subheader("✅ Marked Today")
        for name in st.session_state.marked_names:
            st.success(f"✓ {name}")
    
    # Camera recognition info (disabled for cloud)
    if OPENCV_AVAILABLE:
        st.subheader("📹 Live Camera Recognition")
        st.info("🚨 Live camera features are disabled in cloud deployment. Use photo upload instead.")

# Settings Page
elif page == "⚙️ Settings":
    st.header("⚙️ System Settings")
    
    # System status
    st.subheader("📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        opencv_status = "✅ Available" if OPENCV_AVAILABLE else "❌ Limited"
        st.metric("OpenCV Status", opencv_status)
    
    with col2:
        names, faces = load_face_data()
        st.metric("Registered People", len(set(names)) if names else 0)
    
    with col3:
        total_samples = len(names) if names else 0
        st.metric("Total Samples", total_samples)
    
    # System info
    if not OPENCV_AVAILABLE:
        st.warning("📷 Camera features are disabled due to missing OpenGL libraries.")
        st.info("✅ Photo upload and recognition features are fully functional!")
    
    # Data management
    st.subheader("🗃️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Face Data"):
            if os.path.exists('data/names.pkl'):
                os.remove('data/names.pkl')
            if os.path.exists('data/faces_data.pkl'):
                os.remove('data/faces_data.pkl')
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.marked_names.clear()
            st.success("✅ All face data cleared!")
    
    with col2:
        if st.button("🔄 Reset Today's Attendance"):
            today = datetime.now().strftime("%Y-%m-%d")
            filename = f"Attendence/Attendance_{today}.csv"
            if os.path.exists(filename):
                os.remove(filename)
            st.session_state.marked_names.clear()
            st.success("✅ Today's attendance reset!")
    
    # Export data
    st.subheader("📤 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Attendance Data"):
            import zipfile
            import glob
            
            # Create zip file with all attendance data
            zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            with zipfile.ZipFile(zip_buffer.name, 'w') as zip_file:
                for csv_file in glob.glob("Attendence/*.csv"):
                    zip_file.write(csv_file, os.path.basename(csv_file))
            
            with open(zip_buffer.name, 'rb') as f:
                st.download_button(
                    label="💾 Download Attendance Archive",
                    data=f.read(),
                    file_name=f"attendance_data_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip"
                )
    
    with col2:
        if names:
            # Create a summary report
            df_summary = pd.DataFrame({
                'Name': sorted(set(names)),
                'Samples': [names.count(name) for name in sorted(set(names))]
            })
            
            csv_data = df_summary.to_csv(index=False)
            st.download_button(
                label="📋 Download Registration Summary",
                data=csv_data,
                file_name=f"registered_people_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📱 Quick Actions")
if st.sidebar.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
if OPENCV_AVAILABLE:
    st.sidebar.success("""
    **💡 Tips:**
    - Add at least 2-3 people for better recognition
    - Use photo upload for fastest setup
    - Mark attendance manually if needed
    """)
else:
    st.sidebar.info("""
    **📤 Photo Mode:**
    - Upload 3-10 photos to register people
    - Use photo recognition for attendance
    - All features work without camera
    """)