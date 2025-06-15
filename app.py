import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import time
import os
from datetime import datetime
import uuid
import imageio
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# --- Constants ---
VEHICLE_CLASSES = {
    3: '2-Wheeler',  # Motorcycle
    2: '4-Wheeler',  # Car
    5: '4-Wheeler',  # Bus
    7: '6-Wheeler'   # Truck
}
IMG_SIZE = 640
CONF_THRESHOLD = 0.6  # Higher confidence for better accuracy
IOU_THRESHOLD = 0.7   # Stricter tracking to reduce duplicates
NMS_IOU = 0.7         # NMS IoU to merge overlapping boxes
FRAME_RATE = 30
SLEEP_INTERVAL = 0.03
ANOMALY_THRESHOLD = 1.5
HEATMAP_SIGMA = 20
MAX_STORED_FRAMES = 1000
MIN_FRAMES_FOR_PREDICTION = 60
JAM_OCCUPANCY_THRESHOLD = 30.0
JAM_DISPLACEMENT_THRESHOLD = 5.0
JAM_WINDOW = 10
CONGESTION_PREDICTION_HORIZON = 3600
PIXEL_TO_METER = 0.05  # 1 pixel = 0.05 meters
SPEED_THRESHOLD = 33.33  # 120 km/h = 33.33 m/s

# --- Color Ranges in HSV ---
COLOR_RANGES = {
    'Red': [((0, 100, 50), (10, 255, 255)), ((160, 100, 50), (180, 255, 255))],
    'Blue': [((100, 100, 50), (130, 255, 255))],
    'Green': [((40, 100, 50), (80, 255, 255))],
    'White': [((0, 0, 200), (180, 30, 255))],
    'Black': [((0, 0, 0), (180, 255, 50))],
    'Yellow': [((20, 100, 50), (40, 255, 255))]
}

# --- Language Support ---
LANGUAGES = {
    'en': {
        'title': '🚦 Traffic Analyzer AI 🚦',
        'upload': 'Upload Traffic Video(s)',
        'threshold': 'Set Congestion Threshold',
        'window': 'Analysis Time Window (s)',
        'frame_skip_label': 'Frame Skipping (1x = Process All Frames)',
        'process': 'Start Processing 🚀',
        'success': 'Video {index} Uploaded! ✅ Duration: {duration:.1f} seconds',
        'processing': 'Processing video {index}...',
        'complete': '✅ Processing Completed',
        'download_data': '📥 Download Traffic Data CSV',
        'download_video': '📥 Download Annotated Video (Camera {index})',
        'download_overspeeding': '📥 Download Overspeeding Log CSV',
        'download_matched_vehicles': '📥 Download Matched Vehicles Log CSV',
        'warning': 'Please upload at least one video and start processing.',
        'alert': '⚠️ High Traffic Alert! {count} vehicles detected! 🚨',
        'anomaly': '🚨 Anomaly Detected! Sudden spike in vehicle count: {count}',
        'jam_warning': '🚨 Traffic Jam Warning! High density and low vehicle motion detected!',
        'overspeeding_warning': '🚨 Overspeeding Detected! Vehicle ID {vehicle_id} at {speed:.1f} km/h!',
        'matched_vehicle_alert': '🚨 Matched Vehicle Detected! {vehicle_type} in {color}!',
        'count_title': '📈 Vehicle Count Over Time (All Cameras)',
        'congestion_plot': '🚨 Congestion Spikes (Vehicles > {threshold})',
        'congestion_prediction_title': '🔮 Traffic Congestion Predictions',
        'congestion_prediction_warning': '⚠️ Insufficient data for congestion predictions. Need at least {min_frames} frames.',
        'frame_limit_warning': '⚠️ Maximum frame storage reached ({max_frames}). Older frames discarded. Consider a shorter time window for full video export.',
        'vehicle_type_label': 'Select Vehicle Type to Detect',
        'color_label': 'Select Vehicle Color to Detect'
    },
    'hi': {
        'title': '🚦 ट्रैफिक विश्लेषक AI 🚦',
        'upload': 'ट्रैफिक वीडियो अपलोड करें',
        'threshold': 'भीड़ सीमा निर्धारित करें',
        'window': 'विश्लेषण समय अवधि (सेकंड)',
        'frame_skip_label': 'फ्रेम छोड़ना (1x = सभी फ्रेम प्रोसेस करें)',
        'process': 'प्रसंस्करण शुरू करें 🚀',
        'success': 'वीडियो {index} अपलोड हो गया! ✅ अवधि: {duration:.1f} सेकंड',
        'processing': 'वीडियो {index} प्रसंस्करण हो रहा है...',
        'complete': '✅ प्रसंस्करण पूरा हुआ',
        'download_data': '📥 ट्रैफिक डेटा CSV डाउनलोड करें',
        'download_video': '📥 एनोटेटेड वीडियो डाउनलोड करें (कैमरा {index})',
        'download_overspeeding': '📥 अति गति लॉग CSV डाउनलोड करें',
        'download_matched_vehicles': '📥 मिलान वाहन लॉग CSV डाउनलोड करें',
        'warning': 'कृपया कम से कम एक वीडियो अपलोड करें और प्रसंस्करण शुरू करें।',
        'alert': '⚠️ उच्च ट्रैफिक चेतावनी! {count} वाहन पाए गए! 🚨',
        'anomaly': '🚨 असामान्यता पाई गई! वाहन संख्या में अचानक वृद्धि: {count}',
        'jam_warning': '🚨 ट्रैफिक जाम चेतावनी! उच्च घनत्व और कम वाहन गति पाई गई!',
        'overspeeding_warning': '🚨 अति गति पकड़ी गई! वाहन ID {vehicle_id} {speed:.1f} किमी/घंटा पर!',
        'matched_vehicle_alert': '🚨 मिलान वाहन पकड़ा गया! {vehicle_type} {color} रंग में!',
        'count_title': '📈 समय के साथ वाहन गणना (सभी कैमरे)',
        'congestion_plot': '🚨 भीड़ के शिखर (वाहन > {threshold})',
        'congestion_prediction_title': '🔮 ट्रैफिक भीड़ भविष्यवाणियाँ',
        'congestion_prediction_warning': '⚠️ भीड़ भविष्यवाणियों के लिए अपर्याप्त डेटा। कम से कम {min_frames} फ्रेम चाहिए।',
        'frame_limit_warning': '⚠️ अधिकतम फ्रेम भंडारण सीमा ({max_frames}) पहुंच गई। पुराने फ्रेम हटाए गए। पूर्ण वीडियो निर्यात के लिए छोटी समय अवधि चुनें।',
        'vehicle_type_label': 'पकड़ने के लिए वाहन प्रकार चुनें',
        'color_label': 'पकड़ने के लिए वाहन रंग चुनें'
    }
}

# --- Initialize Session State ---
def init_session_state():
    defaults = {
        "processing_done": False,
        "df": None,
        "video_id": str(uuid.uuid4()),
        "language": "en",
        "annotated_frames": {},
        "frame_limit_reached": False,
        "overspeeding_log": None,
        "matched_vehicle_log": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Get Video Duration ---
def get_video_duration(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
    except Exception:
        return None

# --- Save Uploaded Video ---
def save_uploaded_video(uploaded_video, index):
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_cam{index}.mp4")
        temp_file.write(uploaded_video.read())
        temp_file.close()
        return temp_file.name
    except Exception as e:
        st.error(f"Error saving video {index}: {e}")
        return None

# --- Generate Heatmap ---
def generate_heatmap(frame, boxes, height, width):
    heatmap = np.zeros((height, width), dtype=np.float32)
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        heatmap[y1:y2, x1:x2] += 1
    heatmap = gaussian_filter(heatmap, sigma=HEATMAP_SIGMA)
    heatmap = np.clip(heatmap / heatmap.max() * 255, 0, 255).astype(np.uint8) if heatmap.max() > 0 else heatmap.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Fixed typo: COLMAP_JET -> COLORMAP_JET
    return cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)

# --- Calculate IoU ---
def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# --- Detect Dominant Color ---
def detect_dominant_color(frame, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixel_count = {}
    for color, ranges in COLOR_RANGES.items():
        mask = None
        for lower, upper in ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            color_mask = cv2.inRange(hsv_roi, lower, upper)
            if mask is None:
                mask = color_mask
            else:
                mask = cv2.bitwise_or(mask, color_mask)
        count = cv2.countNonZero(mask)
        pixel_count[color] = count
    if not pixel_count or sum(pixel_count.values()) == 0:
        return None
    dominant_color = max(pixel_count, key=pixel_count.get)
    return dominant_color if pixel_count[dominant_color] > 0 else None

# --- Process Video Frame ---
def process_frame(frame, model, vehicle_paths, vehicle_positions, frame_idx, width, height, prev_boxes, target_vehicle_type, target_color, vehicle_class_history):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=NMS_IOU, verbose=False)
    annotated_frame = results[0].plot()

    count = 0
    type_count = {'2-Wheeler': 0, '4-Wheeler': 0, '6-Wheeler': 0}
    total_box_area = 0
    displacements = []
    current_boxes = []
    overspeeding_count = 0
    matched_vehicle_count = 0
    overspeeding_vehicles = []
    matched_vehicles = []

    # Draw vehicle paths
    for path in vehicle_paths.values():
        if len(path) > 1:
            for i in range(1, len(path)):
                cv2.line(annotated_frame, path[i-1], path[i], (0, 255, 0), 1)

    # Process vehicles with tracking
    new_positions = {}
    new_vehicle_id = max(vehicle_paths.keys(), default=0) + 1
    time_per_frame = 1 / FRAME_RATE
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0]
            current_box = [x1, y1, x2, y2]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Match with previous boxes
            vehicle_id = None
            max_iou = 0
            for prev_id, prev_box in prev_boxes.items():
                iou = calculate_iou(current_box, prev_box)
                if iou > max_iou and iou > IOU_THRESHOLD:
                    max_iou = iou
                    vehicle_id = prev_id

            if vehicle_id is None:
                vehicle_id = new_vehicle_id
                new_vehicle_id += 1

            # Update class history and correct class
            if vehicle_id not in vehicle_class_history:
                vehicle_class_history[vehicle_id] = []
            vehicle_class_history[vehicle_id].append(cls_id)
            if len(vehicle_class_history[vehicle_id]) > 10:  # Keep last 10 frames
                vehicle_class_history[vehicle_id].pop(0)
            majority_cls_id = Counter(vehicle_class_history[vehicle_id]).most_common(1)[0][0]
            vehicle_type = VEHICLE_CLASSES[majority_cls_id]

            count += 1
            type_count[vehicle_type] += 1
            total_box_area += (x2 - x1) * (y2 - y1)
            new_positions[vehicle_id] = (center_x, center_y)
            current_boxes.append((vehicle_id, current_box))

            # Calculate speed
            speed_kmh = None
            if vehicle_id in vehicle_positions:
                old_x, old_y = vehicle_positions[vehicle_id]
                displacement_pixels = ((center_x - old_x)**2 + (center_y - old_y)**2)**0.5
                displacement_meters = displacement_pixels * PIXEL_TO_METER
                speed_ms = displacement_meters / time_per_frame
                speed_kmh = speed_ms * 3.6  # Convert m/s to km/h
                if speed_kmh > SPEED_THRESHOLD * 3.6:
                    overspeeding_count += 1
                    overspeeding_vehicles.append((vehicle_id, speed_kmh))
                    cv2.putText(annotated_frame, f'OVERSPEEDING! {speed_kmh:.1f} km/h',
                                (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Detect color and match vehicle
            if target_vehicle_type and target_color:
                detected_color = detect_dominant_color(frame, box)
                if detected_color == target_color and vehicle_type == target_vehicle_type:
                    matched_vehicle_count += 1
                    matched_vehicles.append((vehicle_id, vehicle_type, detected_color))
                    cv2.putText(annotated_frame, f'MATCHED! {vehicle_type} {detected_color}',
                                (int(x1), int(y1-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Update vehicle path
            if vehicle_id not in vehicle_paths:
                vehicle_paths[vehicle_id] = []
            vehicle_paths[vehicle_id].append((center_x, center_y))
            if len(vehicle_paths[vehicle_id]) > 50:
                vehicle_paths[vehicle_id].pop(0)

    avg_displacement = np.mean(displacements) if displacements else float('inf')

    # Add heatmap
    annotated_frame = generate_heatmap(annotated_frame, results[0].boxes, height, width)

    return annotated_frame, count, type_count, total_box_area, new_positions, avg_displacement, {vid: box for vid, box in current_boxes}, overspeeding_count, matched_vehicle_count, overspeeding_vehicles, matched_vehicles

# --- Create Data Record ---
def create_data_record(frame_idx, elapsed_time, count, type_count, occupancy, camera_index, traffic_jam, overspeeding_count, matched_vehicle_count):
    return {
        'Camera': camera_index,
        'Frame': frame_idx,
        'Time (s)': round(elapsed_time, 2),
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Total Vehicles': count,
        '2-Wheelers': type_count['2-Wheeler'],
        '4-Wheelers': type_count['4-Wheeler'],
        '6-Wheelers': type_count['6-Wheeler'],
        'Occupancy (%)': np.round(occupancy, 2),
        'Traffic Jam': traffic_jam,
        'Overspeeding': overspeeding_count,
        'Matched Vehicle': matched_vehicle_count
    }

# --- Detect Anomalies ---
def detect_anomalies(frame_counts, threshold=ANOMALY_THRESHOLD):
    if len(frame_counts) < 10:
        return []
    mean = np.mean(frame_counts)
    std = np.std(frame_counts)
    return [1 if count > mean + threshold * std else 0 for count in frame_counts]

# --- Random Forest for Congestion Prediction ---
def predict_congestion(df, horizon, frame_rate):
    try:
        df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
        df['Minute'] = pd.to_datetime(df['Timestamp']).dt.minute
        features = df[['Total Vehicles', 'Occupancy (%)', 'Hour', 'Minute', '2-Wheelers', '4-Wheelers', '6-Wheelers']].copy()
        conditions = [
            (df['Occupancy (%)'] <= 20),
            (df['Occupancy (%)'] > 20) & (df['Occupancy (%)'] <= 40),
            (df['Occupancy (%)'] > 40)
        ]
        labels = ['Low', 'Medium', 'High']
        df['Congestion Level'] = np.select(conditions, labels, default='Medium')
        le = LabelEncoder()
        y = le.fit_transform(df['Congestion Level'])
        X = features.values
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        last_time = df['Time (s)'].max()
        last_hour = df['Hour'].iloc[-1]
        last_minute = df['Minute'].iloc[-1]
        future_times = np.linspace(last_time, last_time + horizon, int(horizon * frame_rate // 60))
        future_hours = [(last_hour + (t - last_time) / 3600) % 24 for t in future_times]
        future_minutes = [(last_minute + (t - last_time) / 60) % 60 for t in future_times]
        last_vehicles = df['Total Vehicles'].iloc[-1]
        last_occupancy = df['Occupancy (%)'].iloc[-1]
        last_2w = df['2-Wheelers'].iloc[-1]
        last_4w = df['4-Wheelers'].iloc[-1]
        last_6w = df['6-Wheelers'].iloc[-1]
        future_X = np.array([[last_vehicles, last_occupancy, h, m, last_2w, last_4w, last_6w] for h, m in zip(future_hours, future_minutes)])
        future_y = model.predict(future_X)
        future_probs = model.predict_proba(future_X)
        future_labels = le.inverse_transform(future_y)
        confidence = np.max(future_probs, axis=1)
        return future_times, future_labels, confidence
    except Exception as e:
        st.error(f"Congestion prediction failed: {e}")
        return None, None, None

# --- Plot Congestion Spikes ---
def plot_congestion_spikes(df, congestion_threshold, lang_dict):
    df_cong = df[df['Total Vehicles'] > congestion_threshold]
    if not df_cong.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_cong['Time (s)'], y=df_cong['Total Vehicles'],
            mode='lines+markers', name='Congestion Spikes',
            line=dict(color='crimson', width=2),
            marker=dict(size=8)
        ))
        max_time = df['Time (s)'].max()
        fig.add_trace(go.Scatter(
            x=[0, max_time], y=[congestion_threshold, congestion_threshold],
            mode='lines', name='Threshold',
            line=dict(color='black', dash='dash', width=2)
        ))
        fig.update_layout(
            title=lang_dict['congestion_plot'].format(threshold=congestion_threshold),
            xaxis_title='Time (s)',
            yaxis_title='Vehicle Count',
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        return fig
    return None

# --- Plot Congestion Predictions ---
def plot_congestion_predictions(df, future_times, future_labels, confidence, lang_dict):
    if future_times is None or future_labels is None or confidence is None:
        return None
    last_time = df['Time (s)'].max()
    hist_df = df[['Time (s)', 'Occupancy (%)']].copy()
    hist_df['Congestion Level'] = np.select(
        [(hist_df['Occupancy (%)'] <= 20),
         (hist_df['Occupancy (%)'] > 20) & (hist_df['Occupancy (%)'] <= 40),
         (hist_df['Occupancy (%)'] > 40)],
        ['Low', 'Medium', 'High'],
        default='Medium'
    )
    level_map = {'Low': 1, 'Medium': 2, 'High': 3}
    hist_df['Level Value'] = hist_df['Congestion Level'].map(level_map)
    pred_df = pd.DataFrame({
        'Time (s)': future_times,
        'Congestion Level': future_labels,
        'Level Value': [level_map[l] for l in future_labels],
        'Confidence': confidence
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df['Time (s)'], y=hist_df['Level Value'],
        mode='lines+markers', name='Historical',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        text=hist_df['Congestion Level'],
        hovertemplate='Time: %{x:.2f} s<br>Level: %{text}<br>Value: %{y}'
    ))
    fig.add_trace(go.Scatter(
        x=pred_df['Time (s)'], y=pred_df['Level Value'],
        mode='lines+markers', name='Predicted',
        line=dict(color='orange', width=2, dash='dash'),
        marker=dict(size=6),
        text=pred_df['Congestion Level'],
        hovertemplate='Time: %{x:.2f} s<br>Level: %{text}<br>Confidence: %{customdata:.2%}',
        customdata=pred_df['Confidence']
    ))
    fig.add_trace(go.Scatter(
        x=pred_df['Time (s)'], y=pred_df['Level Value'] + 0.5,
        mode='lines', name='Upper Bound',
        line=dict(color='orange', width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=pred_df['Time (s)'], y=pred_df['Level Value'] - 0.5,
        mode='lines', name='Lower Bound',
        line=dict(color='orange', width=0),
        fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)',
        showlegend=False
    ))

    fig.update_layout(
        title=lang_dict['congestion_prediction_title'],
        xaxis_title='Time (s)',
        yaxis_title='Congestion Level',
        yaxis=dict(
            tickvals=[1, 2, 3],
            ticktext=['Low', 'Medium', 'High']
        ),
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

# --- Plot Interactive Dashboard ---
def plot_interactive_dashboard(df, lang_dict):
    try:
        fig = px.line(df, x='Time (s)', y='Total Vehicles', title=lang_dict['count_title'],
                      labels={'Time (s)': 'Time (s)', 'Total Vehicles': 'Vehicle Count'},
                      color='Camera')
        fig.update_layout(showlegend=True, hovermode='x unified', template='plotly_white')
        return fig
    except Exception as e:
        st.error(f"Error rendering Plotly chart: {e}")
        return None

# --- Save Annotated Video ---
def save_annotated_video(frames, output_path, fps=FRAME_RATE):
    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
        for frame in frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        writer.close()
        return True
    except Exception:
        return False

# --- Process Video ---
def process_video(temp_video_path, model, congestion_threshold, time_window, lang_dict, camera_index, skip_factor, target_vehicle_type, target_color):
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error(f"Failed to open video file {camera_index}.")
        os.unlink(temp_video_path)
        return None, None, None, None

    frame_idx = 0
    actual_frame_idx = 0
    vehicle_paths = {}
    vehicle_positions = {}
    prev_boxes = {}
    vehicle_class_history = {}  # Track class history for consistency
    data_records = []
    overspeeding_log = []
    matched_vehicle_log = []
    frame_counts = []
    frame_times = []
    congestion_flags = []
    anomaly_flags = []
    jam_flags = []
    displacements = []
    occupancies = []
    start_time = time.time()
    annotated_frames = []

    frame_placeholder = st.empty()
    warning_placeholder = st.empty()
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(time_window * FRAME_RATE) if time_window else total_frames

    try:
        while cap.isOpened() and (actual_frame_idx < max_frames or not time_window):
            ret, frame = cap.read()
            if not ret:
                break

            actual_frame_idx += 1
            if actual_frame_idx % skip_factor != 0:
                continue

            frame_idx += 1
            height, width, _ = frame.shape

            # Process frame
            annotated_frame, count, type_count, total_box_area, new_positions, avg_displacement, current_boxes, overspeeding_count, matched_vehicle_count, overspeeding_vehicles, matched_vehicles = process_frame(
                frame, model, vehicle_paths, vehicle_positions, frame_idx, width, height, prev_boxes, target_vehicle_type, target_color, vehicle_class_history
            )
            vehicle_positions = new_positions.copy()
            prev_boxes = current_boxes.copy()

            # Store frames
            if len(annotated_frames) < MAX_STORED_FRAMES:
                annotated_frames.append(annotated_frame.copy())
            else:
                annotated_frames.pop(0)
                annotated_frames.append(annotated_frame.copy())
                st.session_state.frame_limit_reached = True

            # Calculate occupancy
            occupancy = (total_box_area / (width * height)) * 100
            elapsed_time = frame_idx / (FRAME_RATE / skip_factor)

            # Traffic jam detection
            displacements.append(avg_displacement)
            occupancies.append(occupancy)
            if len(displacements) > JAM_WINDOW:
                displacements.pop(0)
                occupancies.pop(0)
            traffic_jam = False
            if len(displacements) == JAM_WINDOW:
                if (np.mean(occupancies) > JAM_OCCUPANCY_THRESHOLD and 
                    np.mean(displacements) < JAM_DISPLACEMENT_THRESHOLD):
                    traffic_jam = True
                    warning_placeholder.error(lang_dict['jam_warning'])
                    cv2.putText(annotated_frame, 'TRAFFIC JAM WARNING!',
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    warning_placeholder.empty()

            # Log overspeeding events
            for vehicle_id, speed_kmh in overspeeding_vehicles:
                overspeeding_log.append({
                    'Camera': camera_index,
                    'Frame': frame_idx,
                    'Time (s)': round(elapsed_time, 2),
                    'Vehicle ID': vehicle_id,
                    'Speed (km/h)': round(speed_kmh, 1)
                })
                warning_placeholder.warning(lang_dict['overspeeding_warning'].format(vehicle_id=vehicle_id, speed=speed_kmh))

            # Log matched vehicle events
            for vehicle_id, vehicle_type, color in matched_vehicles:
                matched_vehicle_log.append({
                    'Camera': camera_index,
                    'Frame': frame_idx,
                    'Time (s)': round(elapsed_time, 2),
                    'Vehicle ID': vehicle_id,
                    'Vehicle Type': vehicle_type,
                    'Color': color
                })
                warning_placeholder.warning(lang_dict['matched_vehicle_alert'].format(vehicle_type=vehicle_type, color=color.lower()))

            # Record data
            record = create_data_record(frame_idx, elapsed_time, count, type_count, occupancy, camera_index, traffic_jam, overspeeding_count, matched_vehicle_count)
            data_records.append(record)
            frame_counts.append(count)
            frame_times.append(elapsed_time)
            congestion_flags.append(1 if count > congestion_threshold else 0)
            jam_flags.append(1 if traffic_jam else 0)

            # Display congestion warning
            if count > congestion_threshold:
                warning_placeholder.warning(lang_dict['alert'].format(count=count))
                cv2.putText(annotated_frame, 'TRAFFIC WARNING: Congestion Detected!',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Update UI
            frame_placeholder.image(annotated_frame, channels="BGR", caption=f'Camera {camera_index} Frame {frame_idx}', use_container_width=True)

            # Update progress
            progress_bar.progress(min(actual_frame_idx / total_frames, 1.0))
            time.sleep(SLEEP_INTERVAL)

    except Exception as e:
        st.error(f"Error processing video {camera_index}: {e}")
        return None, None, None, None
    finally:
        cap.release()
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

    # Create DataFrame
    df = pd.DataFrame(data_records)
    df["Congestion"] = congestion_flags
    df["Traffic Jam"] = jam_flags
    anomaly_flags = detect_anomalies(frame_counts)
    df["Anomaly"] = anomaly_flags

    # Create log DataFrames
    overspeeding_log_df = pd.DataFrame(overspeeding_log)
    matched_vehicle_log_df = pd.DataFrame(matched_vehicle_log)

    return df, annotated_frames, overspeeding_log_df, matched_vehicle_log_df

# --- Main App ---
def main():
    st.set_page_config(layout="wide")
    init_session_state()

    # Language Selection
    lang = st.sidebar.selectbox("Language / भाषा", ["English", "Hindi"])
    st.session_state.language = 'en' if lang == "English" else 'hi'
    lang_dict = LANGUAGES[st.session_state.language]

    st.title(lang_dict['title'])
    
    # Sidebar
    st.sidebar.header(lang_dict['upload'])
    uploaded_videos = st.sidebar.file_uploader(lang_dict['upload'], type=["mp4", "avi", "mov"], accept_multiple_files=True, key=st.session_state.video_id)
    
    # Display Video Durations
    video_paths = []
    if uploaded_videos:
        st.sidebar.write(f"Number of videos uploaded: {len(uploaded_videos)}")
        for index, video in enumerate(uploaded_videos, 1):
            temp_path = save_uploaded_video(video, index)
            if temp_path:
                duration = get_video_duration(temp_path)
                if duration:
                    st.sidebar.success(lang_dict['success'].format(index=index, duration=duration))
                else:
                    st.sidebar.success(lang_dict['success'].format(index=index, duration="Unknown"))
                video_paths.append(temp_path)
            else:
                return

    # Congestion Threshold Presets
    preset = st.sidebar.selectbox("Congestion Preset", ["Custom", "Urban (20)", "Highway (10)"])
    if preset == "Urban (20)":
        congestion_threshold = 20
    elif preset == "Highway (10)":
        congestion_threshold = 10
    else:
        congestion_threshold = st.sidebar.slider(lang_dict['threshold'], min_value=5, max_value=50, value=15)
    
    time_window = st.sidebar.number_input(lang_dict['window'], min_value=0, value=0, step=5, help="Set to 0 for full video analysis")
    skip_factor = st.sidebar.selectbox(lang_dict['frame_skip_label'], [1, 2, 5, 10], format_func=lambda x: f"{x}x")

    # Vehicle Detection Inputs
    st.sidebar.header("Vehicle Detection")
    target_vehicle_type = st.sidebar.selectbox(lang_dict['vehicle_type_label'], ["None", "2-Wheeler", "4-Wheeler", "6-Wheeler"], index=0)
    target_color = st.sidebar.selectbox(lang_dict['color_label'], ["None", "Red", "Blue", "Green", "White", "Black", "Yellow"], index=0)
    if target_vehicle_type == "None" or target_color == "None":
        target_vehicle_type = None
        target_color = None
    
    # Load YOLO model
    try:
        model = YOLO('yolov8x.pt')  # Use extra-large model for higher accuracy
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return

    if uploaded_videos and st.sidebar.button(lang_dict['process']):
        if not video_paths:
            st.error("Video upload failed. Please try again.")
            return
        st.session_state.annotated_frames = {}
        dfs = []
        overspeeding_logs = []
        matched_vehicle_logs = []
        for index, video_path in enumerate(video_paths, 1):
            with st.spinner(lang_dict['processing'].format(index=index)):
                df, frames, overspeeding_log_df, matched_vehicle_log_df = process_video(video_path, model, congestion_threshold, time_window, lang_dict, index, skip_factor, target_vehicle_type, target_color)
                if df is not None:
                    dfs.append(df)
                    st.session_state.annotated_frames[index] = frames
                    if not overspeeding_log_df.empty:
                        overspeeding_logs.append(overspeeding_log_df)
                    if not matched_vehicle_log_df.empty:
                        matched_vehicle_logs.append(matched_vehicle_log_df)
        if dfs:
            st.session_state.df = pd.concat(dfs, ignore_index=True)
            st.session_state.overspeeding_log = pd.concat(overspeeding_logs, ignore_index=True) if overspeeding_logs else pd.DataFrame()
            st.session_state.matched_vehicle_log = pd.concat(matched_vehicle_logs, ignore_index=True) if matched_vehicle_logs else pd.DataFrame()
            st.session_state.processing_done = True

    # Display Results
    if st.session_state.processing_done:
        st.success(lang_dict['complete'])
        df = st.session_state.df

        # Frame Limit Warning
        if st.session_state.frame_limit_reached:
            st.warning(lang_dict['frame_limit_warning'].format(max_frames=MAX_STORED_FRAMES))

        # Interactive Dashboard
        st.subheader(lang_dict['count_title'])
        fig_dashboard = plot_interactive_dashboard(df, lang_dict)
        if fig_dashboard:
            st.plotly_chart(fig_dashboard, use_container_width=True)

        # Congestion Plot
        fig_cong = plot_congestion_spikes(df, congestion_threshold, lang_dict)
        if fig_cong:
            st.plotly_chart(fig_cong, use_container_width=True)

        # Anomaly Alerts
        if df['Anomaly'].sum() > 0:
            anomaly_times = df[df['Anomaly'] == 1][['Time (s)', 'Total Vehicles']]
            for _, row in anomaly_times.iterrows():
                st.warning(lang_dict['anomaly'].format(count=int(row['Total Vehicles'])))

        # Traffic Jam Alerts
        if df['Traffic Jam'].sum() > 0:
            jam_times = df[df['Traffic Jam'] == 1][['Time (s)', 'Occupancy (%)']]
            for _, row in jam_times.iterrows():
                st.error(lang_dict['jam_warning'])

        # Overspeeding Alerts
        if df['Overspeeding'].sum() > 0:
            overspeeding_times = df[df['Overspeeding'] > 0][['Time (s)', 'Overspeeding']]
            for _, row in overspeeding_times.iterrows():
                st.warning(f"Overspeeding events: {int(row['Overspeeding'])} at {row['Time (s)']:.2f}s")

        # Matched Vehicle Alerts
        if df['Matched Vehicle'].sum() > 0 and target_vehicle_type and target_color:
            matched_times = df[df['Matched Vehicle'] > 0][['Time (s)', 'Matched Vehicle']]
            for _, row in matched_times.iterrows():
                st.warning(f"Matched vehicles: {int(row['Matched Vehicle'])} {target_vehicle_type} in {target_color} at {row['Time (s)']:.2f}s")

        # Congestion Predictions
        st.subheader(lang_dict['congestion_prediction_title'])
        if len(df) >= MIN_FRAMES_FOR_PREDICTION:
            future_times, future_labels, confidence = predict_congestion(df, CONGESTION_PREDICTION_HORIZON, FRAME_RATE)
            fig_cong_pred = plot_congestion_predictions(df, future_times, future_labels, confidence, lang_dict)
            if fig_cong_pred:
                st.plotly_chart(fig_cong_pred, use_container_width=True)
        else:
            st.warning(lang_dict['congestion_prediction_warning'].format(min_frames=MIN_FRAMES_FOR_PREDICTION))

        # Download Buttons
        st.download_button(lang_dict['download_data'], data=df.to_csv(index=False),
                          file_name="traffic_data.csv", mime="text/csv")
        
        if not st.session_state.overspeeding_log.empty:
            st.download_button(lang_dict['download_overspeeding'], data=st.session_state.overspeeding_log.to_csv(index=False),
                              file_name="overspeeding_log.csv", mime="text/csv")
        
        if not st.session_state.matched_vehicle_log.empty:
            st.download_button(lang_dict['download_matched_vehicles'], data=st.session_state.matched_vehicle_log.to_csv(index=False),
                              file_name="matched_vehicle_log.csv", mime="text/csv")
        
        for index, frames in st.session_state.annotated_frames.items():
            if frames:
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=f"_cam{index}.mp4").name
                try:
                    if save_annotated_video(frames, temp_output):
                        with open(temp_output, "rb") as f:
                            st.download_button(lang_dict['download_video'].format(index=index), data=f.read(),
                                              file_name=f"annotated_traffic_video_cam{index}.mp4", mime="video/mp4")
                finally:
                    if os.path.exists(temp_output):
                        os.unlink(temp_output)
    else:
        st.warning(lang_dict['warning'])

if __name__ == "__main__":
    main()