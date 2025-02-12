from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import base64
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # Using the smallest model for better speed
model.conf = 0.25  # Increased confidence threshold for fewer detections

# Colors for visualization
BALL_COLOR = (0, 255, 255)  # Yellow for ball (BGR format)
PLAYER_COLOR = (255, 0, 0)  # Blue for players

# Create a thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

def process_frame(frame):
    # Resize frame for faster processing (adjust size as needed)
    scale_percent = 66  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    small_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    
    # Run YOLOv8 inference on the frame
    results = model(small_frame, verbose=False)  # Disable verbose output
    
    # Scale coordinates back to original size
    scale_x = frame.shape[1] / small_frame.shape[1]
    scale_y = frame.shape[0] / small_frame.shape[0]
    
    # Process detections
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            # Scale coordinates back to original size
            x1, y1, x2, y2 = map(int, box.xyxy[0] * [scale_x, scale_y, scale_x, scale_y])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            
            if class_id == 32:  # Sports ball
                # Draw yellow square for ball
                cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
                cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 2)
            
            elif class_id == 0:  # Person
                cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_COLOR, 2)
                cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, PLAYER_COLOR, 2)
    
    # Add timestamp (update less frequently)
    if frame_count % 30 == 0:  # Update timestamp every 30 frames
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

frame_count = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_video_frame():
    global frame_count
    frame_count += 1
    
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    # Read frame from request
    frame_file = request.files['frame']
    frame_bytes = frame_file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Invalid frame data'}), 400
    
    # Process the frame
    processed_frame = process_frame(frame)
    
    # Convert processed frame back to JPEG with lower quality for faster transmission
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
    _, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'processed_frame': f'data:image/jpeg;base64,{frame_base64}'
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True)