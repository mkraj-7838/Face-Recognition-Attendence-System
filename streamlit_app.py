import streamlit as st
import cv2
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

# Initialize face detector
@st.cache_resource
def load_face_detector():
    if not os.path.exists('data/face.xml'):
        st.error("❌ face.xml not found! Please run setup first.")
        return None
    return cv2.CascadeClassifier('data/face.xml')

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

def generate_face_variations(face_image):
    """Generate multiple variations from a single face image for better training"""
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
    for alpha in [0.8, 1.2, 0.9, 1.1]:  # brightness factor
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
    
    # Check if camera is available
    names, faces = load_face_data()
    st.info(f"Currently registered: {len(set(names))} people with {len(names)} total samples")
    st.success("🚀 **New Fast Mode**: Camera capture now takes only 20 samples (~30-60 seconds) instead of 100!")
    st.info("📤 **Photo Upload**: Upload 3-10 photos for even faster registration!")
    
    # Input for person's name
    person_name = st.text_input("👤 Enter person's name:", key="person_name")
    
    if person_name:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📹 Start Camera", key="start_camera"):
                st.session_state.camera_active = True
        
        with col2:
            if st.button("⏹️ Stop Camera", key="stop_camera"):
                st.session_state.camera_active = False
        
        # Camera capture section
        if st.session_state.camera_active:
            st.subheader("📸 Face Capture")
            
            # Instructions
            # Choose capture method
            capture_method = st.radio(
                "Choose capture method:",
                ["📹 Live Camera (20 samples)", "📤 Upload Photos"],
                key="capture_method"
            )
            
            if capture_method == "📤 Upload Photos":
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
                        facedetect = load_face_detector()
                        if facedetect is None:
                            st.error("❌ Face detector not loaded!")
                            st.stop()
                        
                        faces_data = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, uploaded_file in enumerate(uploaded_files):
                            # Read uploaded image
                            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                            
                            if image is None:
                                st.warning(f"⚠️ Could not process {uploaded_file.name}")
                                continue
                            
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            faces = facedetect.detectMultiScale(gray, 1.3, 5)
                            
                            faces_found = 0
                            for (x, y, w, h) in faces:
                                if faces_found >= 1:  # Only take first face from each photo
                                    break
                                    
                                crop_img = image[y:y+h, x:x+w]
                                crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                                
                                # Generate multiple variations from single photo
                                variations = generate_face_variations(crop_img_gray)
                                faces_data.extend(variations)
                                faces_found += 1
                            
                            if faces_found == 0:
                                st.warning(f"⚠️ No face detected in {uploaded_file.name}")
                            
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
                            st.error(f"❌ Need at least 15 samples. Only got {len(faces_data)}. Try uploading more photos or use camera capture.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing photos: {str(e)}")
                        
            else:  # Live Camera
                st.info("""
                📋 **Camera Instructions:**
                1. Position your face in the camera view
                2. The system will automatically capture 20 samples (much faster!)
                3. Move your head slightly for better variety
                4. Takes about 30-60 seconds instead of 5+ minutes
                """)
            
            # Placeholder for camera feed and progress
            camera_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize camera
            try:
                cap = cv2.VideoCapture(0)
                facedetect = load_face_detector()
                
                if facedetect is None:
                    st.error("❌ Face detector not loaded!")
                    st.stop()
                
                faces_data = []
                frame_count = 0
                
                while len(faces_data) < 20 and st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Could not read from camera")
                        break
                    
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = facedetect.detectMultiScale(gray, 1.3, 5)
                    
                    # Process faces
                    for (x, y, w, h) in faces:
                        if len(faces_data) < 20 and frame_count % 5 == 0:  # Capture every 5th frame instead of 10th
                            crop_img = frame[y:y+h, x:x+w]
                            crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                            resized_img = cv2.resize(crop_img_gray, (50, 50))
                            faces_data.append(resized_img)
                        
                        # Draw rectangle around face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Samples: {len(faces_data)}/20", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    frame_count += 1
                    
                    # Update progress
                    progress = len(faces_data) / 20
                    progress_bar.progress(progress)
                    status_text.text(f"Captured: {len(faces_data)}/20 samples")
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    time.sleep(0.05)  # Faster capture rate
                
                # Save collected data and generate variations
                if len(faces_data) >= 15:  # Need at least 15 samples
                    # Generate additional variations from captured samples
                    enhanced_faces_data = []
                    for face in faces_data:
                        variations = generate_face_variations(face)
                        enhanced_faces_data.extend(variations[:3])  # Take first 3 variations per sample
                    
                    # Load existing data
                    existing_names, existing_faces = load_face_data()
                    
                    # Add new data
                    total_samples = len(enhanced_faces_data)
                    new_names = existing_names + [person_name] * total_samples
                    new_faces = existing_faces + enhanced_faces_data
                    
                    # Save updated data
                    save_face_data(new_names, new_faces)
                    
                    st.success(f"✅ Successfully added {person_name} with {total_samples} face samples (enhanced from {len(faces_data)} captures)!")
                    st.balloons()
                    
                    # Clear cache to retrain model
                    st.cache_resource.clear()
                else:
                    st.warning(f"⚠️ Only captured {len(faces_data)} samples. Need at least 15. Try again.")
                
                cap.release()
                st.session_state.camera_active = False
                
            except Exception as e:
                st.error(f"❌ Camera error: {str(e)}")
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
            # Add to attendance
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
    
    # Camera-based attendance
    st.subheader("📹 Camera-based Attendance")
    
    col1, col2 = st.columns(2)
    with col1:
        start_recognition = st.button("📹 Start Recognition")
    with col2:
        stop_recognition = st.button("⏹️ Stop Recognition")
    
    if start_recognition:
        st.session_state.camera_active = True
    if stop_recognition:
        st.session_state.camera_active = False
    
    # Display marked attendance
    if st.session_state.marked_names:
        st.subheader("✅ Marked Today")
        for name in st.session_state.marked_names:
            st.success(f"✓ {name}")
    
    # Camera recognition
    if st.session_state.camera_active:
        st.info("📹 Camera is active. Face will be recognized automatically.")
        st.info("🔄 Recognition runs every few seconds to avoid duplicates.")
        
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            cap = cv2.VideoCapture(0)
            facedetect = load_face_detector()
            
            frame_count = 0
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Could not read from camera")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, 1.3, 5)
                
                current_time = datetime.now()
                date_str = current_time.strftime('%Y-%m-%d')
                time_str = current_time.strftime('%H:%M:%S')
                filename = f"Attendence/Attendance_{date_str}.csv"
                
                recognized_names = []
                
                for (x, y, w, h) in faces:
                    # Recognize face every 30 frames to avoid spam
                    if frame_count % 30 == 0:
                        crop_img = frame[y:y+h, x:x+w]
                        crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        resized_img = cv2.resize(crop_img_gray, (50, 50)).flatten().reshape(1, -1)
                        
                        try:
                            pred = knn.predict(resized_img)
                            name = str(pred[0])
                            recognized_names.append(name)
                            
                            # Auto-mark attendance if not already marked today
                            if name not in st.session_state.marked_names:
                                file_exists = os.path.isfile(filename)
                                with open(filename, "a", newline='') as csvfile:
                                    writer = csv.writer(csvfile)
                                    if not file_exists:
                                        writer.writerow(['Name', 'Date', 'Time'])
                                    writer.writerow([name, date_str, time_str])
                                
                                st.session_state.marked_names.add(name)
                                status_placeholder.success(f"✅ Attendance marked for {name}!")
                        except Exception as e:
                            name = "Unknown"
                    else:
                        name = "Detecting..."
                    
                    # Draw rectangle and name
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                frame_count += 1
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                time.sleep(0.1)
            
            cap.release()
            
        except Exception as e:
            st.error(f"❌ Camera error: {str(e)}")
            st.session_state.camera_active = False

# Settings Page
elif page == "⚙️ Settings":
    st.header("⚙️ System Settings")
    
    # System status
    st.subheader("📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        face_xml_exists = os.path.exists('data/face.xml')
        st.metric("Face Detector", "✅ Ready" if face_xml_exists else "❌ Missing")
    
    with col2:
        names, faces = load_face_data()
        st.metric("Registered People", len(set(names)) if names else 0)
    
    with col3:
        total_samples = len(names) if names else 0
        st.metric("Total Samples", total_samples)
    
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
    
    # Download face cascade
    st.subheader("⬇️ Setup")
    if not face_xml_exists:
        st.warning("❌ face.xml not found!")
        st.info("""
        To set up the face detector:
        1. Download haarcascade_frontalface_default.xml from OpenCV
        2. Rename it to face.xml
        3. Place it in the data/ folder
        """)
    
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
st.sidebar.info("""
**💡 Tips:**
- Add at least 2-3 people for better recognition
- Ensure good lighting when capturing faces
- Mark attendance manually if camera recognition fails
""")
