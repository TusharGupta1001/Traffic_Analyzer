# 🚦 Traffic Analyzer AI

A smart, real-time traffic monitoring web app that detects, tracks, and analyzes traffic using AI! Upload your traffic videos, and the app detects vehicles, tracks their movement, identifies congestion, flags overspeeding vehicles, and even finds cars of a specific **type** and **color**. Oh—and yes, it also gives you **cool graphs** and **downloadable reports**!

---

## 🚗 **Core Capabilities:**

- **Vehicle Detection & Tracking:** Identifies 2-wheelers, 4-wheelers, and 6-wheelers using YOLOv8, tracking them in motion with IoU-based trajectory mapping.
- **Speed & Violation Alerts:** Flags speed violations over 120 km/h and logs overspeed events with timestamped snapshots.
- **Custom Search:** Detects user-specified vehicles (like "red 4-wheelers") via HSV color filtering.
- **Congestion & Jam Detection:** Spots traffic jams when density exceeds 30% and movement drops below 5 pixels/frame.
- **Live Heatmaps:** Generates dynamic occupancy heatmaps to visualize high-traffic zones in real-time.
- **Traffic Forecasting:** Predicts congestion levels (Low/Medium/High) for the next hour using a trained Random Forest model.
- **Trend Detection:** Issues alerts when vehicle counts spike beyond 1.5 standard deviations from the norm.
- **Multi-Camera Handling:** Simultaneously analyzes multiple feeds with distinct stats and visualizations.
- **Interactive Visuals:** Stunning Plotly graphs for time-based trends, counts, and alerts.
- **Smart Logging:** Automatically exports CSVs of detailed logs and saves annotated videos.
- **Customization:** User controls for detection thresholds, frame limits, and UI toggles.
- **Failsafe Engine:** Intelligent exception handling, frame caps (max 1000), and memory-safe operations.

---

## 💡 What Does It Do?

Think of it like a traffic officer with AI superpowers. Here's what it does:

- 🚗 **Detects Vehicles:** Cars, trucks, bikes—you name it.
- 🎯 **Tracks Vehicles:** Follows them across frames.
- 📊 **Counts & Classifies:** Tells you how many 2-wheelers, 4-wheelers, etc.
- 🚨 **Detects Congestion & Jams:** Spots slow-moving clusters of vehicles.
- ⚡ **Flags Overspeeding:** Catches speedsters going above 120 km/h.
- 🎯 **Matches Specific Vehicles:** Looking for red trucks? It finds them!
- 📉 **Predicts Congestion:** Forecasts traffic using Machine Learning.
- 🌡️ **Generates Heatmaps:** Shows vehicle density hotspots.
- 📥 **Exports CSVs & Annotated Videos:** All reports and results are downloadable.

---

## 🛠️ Tools & Libraries Used

| Tool/Library     | Purpose |
|------------------|---------|
| **Streamlit**    | Web interface |
| **OpenCV**       | Frame processing, drawing |
| **YOLOv8 (Ultralytics)** | Vehicle detection |
| **NumPy**        | Math and arrays |
| **Pandas**       | DataFrames and CSV export |
| **Plotly**       | Interactive graphs |
| **ImageIO**      | Video output |
| **SciPy**        | Heatmap smoothing |
| **Scikit-learn** | Congestion prediction |
| **Tempfile**     | Temporary file storage |
| **OS, Time, UUID, Datetime, Collections** | System utilities |

---

## 🔍 How It Works: A Breakdown

1. **Upload Video:** MP4, AVI, MOV supported.
2. **Frame-by-Frame Analysis:**
   - YOLOv8 detects vehicles with high confidence.
   - Vehicle speeds are calculated.
   - Colors are matched using HSV ranges.
3. **Tracking:** Smart tracking with bounding boxes + path lines.
4. **Congestion & Jam Detection:**
   - Based on occupancy % and motion levels.
   - Detected using thresholds and rolling analysis.
5. **ML Prediction:** Random Forest predicts congestion for the next hour.
6. **Output:** Annotated video with overlays, CSV logs, interactive charts.

---

## 📈 Visualizations Included

- Vehicle Count vs. Time
- Traffic Jam Alerts
- Overspeeding Logs
- Congestion Level Prediction
- Live Heatmaps (Red = busy!)

---
## Traffic Analyzer AI Use Cases

### 1. **Vehicle Detection & Tracking**
   **Use Case:**  
   **Urban Traffic Surveillance**:  
   Urban traffic management centers can use Traffic Analyzer AI to detect and track all vehicles on highways and major roads in real-time, ensuring efficient monitoring of traffic flow and helping reduce congestion by identifying the root causes of bottlenecks.

---

### 2. **Speed & Violation Alerts**
   **Use Case:**  
   **Speed Enforcement on Highways**:  
   Law enforcement agencies can use Traffic Analyzer AI to detect speeding vehicles on highways. The system will automatically flag vehicles traveling above 120 km/h, helping authorities issue traffic fines more efficiently and ensuring better road safety.

---

### 3. **Custom Search for Specific Vehicles**
   **Use Case:**  
   **VIP or Emergency Vehicle Identification**:  
   Emergency services, such as ambulances or fire trucks, can be monitored by Traffic Analyzer AI to ensure clear and fast routes in dense traffic. The system can detect user-specified vehicles (e.g., red 4-wheelers) and highlight their paths for priority clearance.

---

### 4. **Congestion & Jam Detection**
   **Use Case:**  
   **Smart City Infrastructure Optimization**:  
   In smart city initiatives, Traffic Analyzer AI can automatically identify traffic jams based on vehicle density and motion patterns, providing city planners with live congestion data. This information can be used to trigger traffic signal adjustments or rerouting to prevent gridlocks.

---

### 5. **Live Heatmaps**
   **Use Case:**  
   **Real-Time Traffic Monitoring for City Planners**:  
   City traffic planners can use the heatmaps generated by Traffic Analyzer AI to visualize areas of high traffic congestion in real-time. This helps with resource allocation and planning for infrastructure improvements, such as adding lanes or redesigning intersections.

---

### 6. **Traffic Forecasting**
   **Use Case:**  
   **Traffic Demand Prediction for Public Transport Systems**:  
   Public transport authorities can use Traffic Analyzer AI's forecasting capabilities to predict future congestion (low, medium, or high) based on real-time traffic data. This can help optimize bus and train schedules during peak times and ensure efficient public transport flow.

---

### 7. **Trend Detection and Alerts**
   **Use Case:**  
   **Anomaly Detection for Traffic Patterns**:  
   In cities with fluctuating traffic conditions, Traffic Analyzer AI can identify unusual spikes in vehicle counts, triggering alerts when there are 1.5 standard deviations above the normal traffic. This could signal accidents, roadworks, or weather conditions affecting traffic patterns.

---

### 8. **Multi-Camera Handling**
   **Use Case:**  
   **Large-Scale Traffic Surveillance Networks**:  
   Large cities or industrial areas with multiple traffic monitoring cameras can leverage Traffic Analyzer AI's multi-camera handling to analyze data from different locations simultaneously. This consolidated data helps give a broader view of traffic conditions across a city or highway system.

---

### 9. **Interactive Visualizations**
   **Use Case:**  
   **Traffic Reporting for Government & Stakeholders**:  
   Traffic analysts and government bodies can generate interactive, visual traffic reports using Plotly graphs. These reports can showcase real-time traffic trends and provide clear, visual summaries of vehicle counts, speed violations, and congestion levels for stakeholders and policymakers.

---

### 10. **Smart Logging and Reporting**
   **Use Case:**  
   **Automated Traffic Data Archiving for Research**:  
   Researchers and urban planners can use Traffic Analyzer AI to automatically log traffic data over extended periods. This data, saved as CSVs and annotated videos, can be archived for historical analysis or future studies on the effectiveness of traffic management policies and infrastructure projects.

---

## 📂 What Can You Download?

✅ Annotated Video (MP4)  
✅ Traffic Report CSV  
✅ Overspeeding Log  
✅ Vehicle Match Log

---

## ⚙️ Technologies & Stack

| Layer | Stack |
|-------|-------|
| UI    | Streamlit |
| AI Model | YOLOv8 (Ultralytics) |
| ML Prediction | Scikit-learn (Random Forest) |
| Video Processing | OpenCV, NumPy |
| Graphs | Plotly |
| Output Handling | Pandas, ImageIO |
| Data Smoothing | SciPy |
| Language Toggle | LangDict based |


---

## 📈 Visuals You Get

- 📊 Vehicle Count Graph (vs Time)
- 🔴 Congestion Alerts Timeline
- ⚡ Speed Distribution
- 🌡️ Real-Time Heatmap
- 📌 Detected Vehicle Snapshot
- 🗃️ CSV Reports + MP4 Annotated Videos

---

## 🔧 Benchmarks (on i5, 8GB RAM)

| Task | Avg Speed |
|------|-----------|
| Frame Processing | ~15 FPS |
| Vehicle Detection | ~0.08s/frame |
| Video Export | ~1.2x realtime |
| CSV Report Export | Instant |
| Forecast ML Model | ~1s per prediction |

---

## 🧪 Performance Tweaks

- 🟢 Confidence threshold tuning
- 🟢 Frame skipping for faster speed
- 🟢 Detection line toggling
- 🟢 Temporary vehicle tracking via UUID
- 🟢 Deprecation warnings suppressed

---

## ❓ FAQs

**Q1: Can I use my own videos?**  
✅ Yes! Just upload traffic videos from your system and let Traffic Analyzer AI work its magic.

---

**Q2: Is this system real-time?**  
🕐 While it works on pre-recorded videos, the processing speed is fast enough to provide near-live insights with minimal delay.

---

**Q3: What happens to uploaded data?**  
🗑️ All data is processed locally and deleted from temporary storage after use to ensure privacy and security.

---

**Q4: What types of vehicles can be detected?**  
🚗🚲🚚 The system detects various vehicles, including cars, bikes, trucks, and even identifies vehicle types like 2-wheelers, 4-wheelers, and 6-wheelers.

---

**Q5: How does the vehicle speed detection work?**  
⚡ It uses advanced motion analysis and vehicle tracking to detect when a vehicle crosses the 120 km/h threshold, automatically flagging speed violations.

---

**Q6: Can I customize the vehicle search?**  
🔍 Absolutely! You can specify particular vehicles, such as red 4-wheelers, using custom HSV color analysis.

---

**Q7: How accurate are the traffic jam detections?**  
🚥 Traffic jams are flagged based on real-time vehicle density and movement, with a threshold set at 30% density and motion below 5 pixels. It’s pretty accurate in real-world conditions!

---

**Q8: What is the congestion forecasting feature?**  
🔮 The AI predicts the traffic congestion (Low, Medium, High) for the next hour based on current traffic trends, using a trained Random Forest model.

---

**Q9: Can I analyze data from multiple cameras?**  
📷 Yes! Traffic Analyzer AI can handle multiple camera feeds simultaneously, providing a comprehensive view of the traffic situation across different areas.

---

**Q10: How do I export the data?**  
💾 You can export traffic data, including vehicle counts and logs, as CSV files for further analysis, and even save annotated videos with vehicle paths and traffic violations.

---


## 🎬 Demo Video

[![Watch the Demo Video](https://img.youtube.com/vi/6qNhkllHpeg/0.jpg)](https://youtu.be/6qNhkllHpeg)

👉 Click the image above or [watch the video on YouTube](https://youtu.be/6qNhkllHpeg)

---
