from PIL import Image
import numpy as np
import base64
from flask import Flask, render_template, request, Response, jsonify
import os
import cv2
from ultralytics import YOLO
import requests
from datetime import datetime
from werkzeug.utils import secure_filename
from io import BytesIO
import threading
import uuid
from pymongo import MongoClient
from bson import ObjectId
import time

# Global progress store
video_tasks = {}

# ---------------- MONGODB SETUP ----------------
# Standard local MongoDB connection
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    db = client["drones_db"]
    detections_collection = db["detections"]
    # Test connection
    client.server_info()
    print(" * Connected to MongoDB successfully")
except Exception as e:
    print(f" * Could not connect to MongoDB: {e}")
    detections_collection = None

# Cooldown for saving detections (in seconds)
SAVE_COOLDOWN = 5
last_save_time = 0

model_lock = threading.Lock()

# ---------------- FLASK APP ----------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
HISTORY_FOLDER = "static/detections"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)

model = YOLO("best.pt")

# Rename class name for display ONLY (if class 6 exists)
if len(model.names) > 6:
    model.names[6] = "Commercial Drone"



# ---------------- CONFIDENCE THRESHOLD ----------------
# Reduce for better detection testing
CONF_THRESHOLD = 0.40

# ---------------- LIVE ALERT STATUS ----------------
drone_alert_status = {
    "detected": False,
    "drone_name": "Drone",
    "location": "Waiting for GPS...",
    "time": ""
}

current_gps = {
    "latitude": None,
    "longitude": None,
    "location_details": "Waiting for GPS..."
}

# ---------------- REVERSE GEO LOCATION ----------------
def get_location_details(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {"User-Agent": "DroneDetectionSystem/1.0"}

        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()

        if "address" in data:
            addr = data["address"]
            city = addr.get("city") or addr.get("town") or addr.get("village") or "Unknown"
            state = addr.get("state", "")
            country = addr.get("country", "")

            location = city
            if state:
                location += f", {state}"
            if country:
                location += f", {country}"

            return location
    except Exception as e:
        print(f"Error getting location: {e}")

    return f"Lat: {lat}, Lon: {lon}"

# ---------------- SAVE TO DATABASE ----------------
def save_detection_to_db(frame, drone_name, confidence, location, lat, lon):
    global last_save_time
    
    if detections_collection is None:
        return

    current_time = time.time()
    if current_time - last_save_time < SAVE_COOLDOWN:
        return

    try:
        # Generate filename
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{drone_name.replace(' ', '_')}.jpg"
        filepath = os.path.join(HISTORY_FOLDER, filename)
        
        # Save image
        cv2.imwrite(filepath, frame)
        
        # Relative path for web serving
        web_path = f"/static/detections/{filename}"
        
        # Prepare record
        record = {
            "drone_name": drone_name,
            "confidence": float(confidence),
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": web_path,
            "created_at": datetime.now()
        }
        
        # Insert into MongoDB
        detections_collection.insert_one(record)
        last_save_time = current_time
        print(f" * Detection saved to MongoDB: {drone_name}")
        
    except Exception as e:
        print(f" * Error saving detection: {e}")

# ---------------- DRAW ALL TRAINED CLASSES ----------------
def draw_detections(frame, results, save_to_db=True):
    global drone_alert_status

    if not results or len(results[0].boxes) == 0:
        drone_alert_status["detected"] = False
        return frame, "No Drone Detected"

    h, w, _ = frame.shape
    drone_found = False
    detected_name = ""

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        bw = x2 - x1
        bh = y2 - y1
        box_area = bw * bh
        frame_area = h * w

        aspect_ratio = bw / (bh + 1e-6)

        # ❌ REMOVE AIRPLANES
        # if box_area > frame_area * 0.30:
        #     continue

        # if aspect_ratio > 3.0:
        #     continue

        class_name = model.names.get(cls_id, "Drone")

        # If model returned a generic 'drone' label, treat it as Commercial Drone
        if isinstance(class_name, str) and class_name.strip().lower() == "drone":
            class_name = "Commercial Drone"

        label = f"{class_name} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0, 255, 0), 3)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        drone_found = True
        detected_name = class_name

    if drone_found:
        drone_alert_status["detected"] = True
        drone_alert_status["drone_name"] = detected_name
        drone_alert_status["location"] = current_gps["location_details"]
        # Save to DB if confidence is high enough and save_to_db is True
        if save_to_db:
            save_detection_to_db(frame, detected_name, results[0].boxes[0].conf[0], 
                                 drone_alert_status["location"], 
                                 current_gps.get("latitude"), 
                                 current_gps.get("longitude"))

        # Return the detected class name (templates will format the display)
        return frame, detected_name

    drone_alert_status["detected"] = False
    return frame, "No Drone Detected"


# ---------------- HOME ----------------
@app.route("/")
def index():
    return render_template("index.html", datetime_now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ---------------- IMAGE DETECTION ----------------
@app.route("/detect_image", methods=["POST"])
def detect_image():
    image = request.files.get("image")
    if not image:
        return render_template("index.html")

    pil_image = Image.open(image.stream).convert("RGB")
    img_array = np.array(pil_image)

    results = model(img_array, conf=CONF_THRESHOLD)
    annotated, detected_label = draw_detections(img_array.copy(), results)

    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return render_template(
        "index.html",
        image_base64=img_base64,
        # Pass plain detected class name to template (e.g. 'Commercial Drone')
        drone_type=detected_label,
        active_section="image",
        datetime_now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

# ---------------- VIDEO DETECTION ----------------
@app.route("/detect_video", methods=["POST"])
def detect_video():
    video = request.files.get("video")
    if not video:
        return render_template("index.html")

    video_filename = secure_filename(video.filename)
    if not video_filename:
        video_filename = "video.mp4"
    import uuid
    video_filename = f"{uuid.uuid4().hex}_{video_filename}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    video.save(video_path)
    
    # Task Tracking
    task_id = request.form.get("task_id", str(uuid.uuid4()))
    video_tasks[task_id] = {"progress": 0, "status": "processing"}

    output_video = "output_" + video_filename.rsplit(".", 1)[0] + ".webm"
    output_path = os.path.join(UPLOAD_FOLDER, output_video)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    
    # Use WebM (VP8) for high browser compatibility and to avoid libopenh264 issues
    try:
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            print("VP80 codec failed, falling back to XVID/AVI (experimental)")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = output_path.replace(".webm", ".avi")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    except Exception as e:
        print(f"Video writer init error: {e}")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace(".webm", ".avi")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"CRITICAL Error: Could not open video writer for {output_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        results = model(frame, conf=CONF_THRESHOLD)
        # ❌ Do not store video detections in DB to avoid bloat
        frame, _ = draw_detections(frame, results, save_to_db=False)
        writer.write(frame)
        
        # Update progress sparingly (every 5 frames)
        if frame_count % 5 == 0:
            video_tasks[task_id]["progress"] = int((frame_count / total_frames) * 100)

    cap.release()
    writer.release()
    video_tasks[task_id]["progress"] = 100
    video_tasks[task_id]["status"] = "finished"
    
    # Verify file was created and has content
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"Output video created: {output_path} ({size} bytes)")
    else:
        print(f"Error: Output file {output_path} was NOT created.")

    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception as e:
        print(f"Error removing {video_path}: {e}")

    # Add timestamp to bypass browser cache
    video_url = "/" + output_path.replace("\\", "/") + f"?t={int(datetime.now().timestamp())}"
    video_tasks[task_id]["result_url"] = video_url

    # Check if AJAX request (JSON response)
    if request.headers.get('Accept') == 'application/json':
        return jsonify({
            "status": "success",
            "video_url": video_url,
            "drone_type": "Processed Result"
        })

    return render_template(
        "index.html",
        video_result=video_url,
        video_url=video_url,
        is_stream=False,
        active_section="video",
        datetime_now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route("/video_status/<task_id>")
def video_status(task_id):
    if task_id in video_tasks:
        return jsonify(video_tasks[task_id])
    # Fallback for when the poll starts before the POST request fully initializes the task
    return jsonify({"progress": 0, "status": "initializing"})

@app.route("/detect_frame", methods=["POST"])
def detect_frame():
    try:
        data = request.json
        img_data = data.get("image")
        if not img_data:
            return jsonify({"error": "No image data"}), 400

        header, encoded = img_data.split(",", 1)
        decoded = base64.b64decode(encoded)
        image = Image.open(BytesIO(decoded)).convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with model_lock:
            results = model(frame, conf=CONF_THRESHOLD)
            
        annotated, detected_label = draw_detections(frame, results)

        _, buffer = cv2.imencode(".jpg", annotated)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        response = {
            "image": "data:image/jpeg;base64," + img_base64,
            "detected": drone_alert_status["detected"],
            "drone_name": drone_alert_status["drone_name"],
            "location": drone_alert_status["location"],
            "time": drone_alert_status["time"],
            "latitude": current_gps.get("latitude"),
            "longitude": current_gps.get("longitude")
        }
        return jsonify(response)
    except Exception as e:
        print(f"Error during frame detection: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------- LIVE CAMERA STREAM (SERVER SIDE) ----------------
def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 🔥 Minor speed improvement (lower resolution)
    cap.set(3, 640)   # width
    cap.set(4, 480)   # height

    if not cap.isOpened():
        return

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # 🔥 Skip every alternate frame (minor speed boost)
        if frame_count % 2 != 0:
            continue

        with model_lock:
            results = model(
                frame,
                conf=CONF_THRESHOLD,
                imgsz=480,          # smaller input size = faster
                iou=0.4,
                agnostic_nms=True,
                verbose=False
            )

        frame, _ = draw_detections(frame, results)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/live_page")
def live_page():
    return render_template("live.html")

@app.route("/drone_status")
def drone_status():
    return jsonify(drone_alert_status)

@app.route("/update_location", methods=["POST"])
def update_location():
    global current_gps
    data = request.json

    lat = float(data.get("latitude"))
    lon = float(data.get("longitude"))

    location_details = get_location_details(lat, lon)

    current_gps["latitude"] = lat
    current_gps["longitude"] = lon
    current_gps["location_details"] = location_details

    return {"status": "ok", "location": location_details}

# ---------------- HISTORY ROUTES ----------------
@app.route("/history")
def history():
    if detections_collection is None:
        return "MongoDB is not connected. Please check your setup.", 500
        
    # Fetch detections sorted by most recent first
    detections = list(detections_collection.find().sort("created_at", -1))
    return render_template("history.html", detections=detections)

@app.route("/delete_history/<id>", methods=["POST"])
def delete_history(id):
    if detections_collection is None:
        return jsonify({"error": "DB not connected"}), 500
        
    try:
        # Find record to get image path
        record = detections_collection.find_one({"_id": ObjectId(id)})
        if record and "image_path" in record:
            # Delete image file
            img_path = record["image_path"].lstrip("/")
            if os.path.exists(img_path):
                os.remove(img_path)
        
        # Delete from DB
        detections_collection.delete_one({"_id": ObjectId(id)})
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if detections_collection is None:
        return jsonify({"error": "DB not connected"}), 500
        
    try:
        # Delete all images
        for filename in os.listdir(HISTORY_FOLDER):
            file_path = os.path.join(HISTORY_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        # Clear collection
        detections_collection.delete_many({})
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    from pyngrok import ngrok, conf
    
    # Setup ngrok
    ngrok_token = os.environ.get("NGROK_AUTH_TOKEN", "3A0k9W59g2CxPol4BY7ONn3zKU1_7yVzKSpFX2hUcN8nRdU3b")
    conf.get_default().auth_token = ngrok_token
    try:
        public_url = ngrok.connect(5000, domain="uncurbed-roxane-thermoscopical.ngrok-free.dev").public_url
        print(f" * ngrok tunnel available at {public_url}")
    except Exception as e:
        print(f" * Failed to start ngrok: {e}")

    app.run(debug=True, threaded=True, use_reloader=False)
