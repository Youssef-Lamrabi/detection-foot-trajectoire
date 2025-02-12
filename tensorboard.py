from flask import Flask, request, Response, render_template, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import base64
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
model = YOLO('yolov8n.pt')

# Initialize TensorBoard writer
log_dir = os.path.join('runs', 'live', datetime.now().strftime('%Y%m%d-%H%M%S'))
writer = SummaryWriter(log_dir)

# Initialize thermal matrix
cumulative_heat_matrix = np.zeros((32, 32))
frame_count = 0

def create_thermal_matrix(detections, image_size=(640, 480), grid_size=32):
    """Create a thermal matrix based on detection densities"""
    heat_matrix = np.zeros((grid_size, grid_size))
    cell_width = image_size[0] // grid_size
    cell_height = image_size[1] // grid_size
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Convert bbox coordinates to grid cells
        grid_x1 = min(int(x1 / cell_width), grid_size-1)
        grid_y1 = min(int(y1 / cell_height), grid_size-1)
        grid_x2 = min(int(x2 / cell_width), grid_size-1)
        grid_y2 = min(int(y2 / cell_height), grid_size-1)
        
        # Add weighted detection to heat matrix
        heat_matrix[grid_y1:grid_y2+1, grid_x1:grid_x2+1] += confidence
    
    return heat_matrix

def plot_thermal_matrix(heat_matrix):
    """Create a thermal matrix plot"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(heat_matrix, 
                cmap='inferno',
                annot=False,
                fmt='.2f',
                cbar_kws={'label': 'Detection Density'})
    plt.title('Live Detection Density Heatmap')
    plt.xlabel('X Grid')
    plt.ylabel('Y Grid')
    return plt.gcf()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global cumulative_heat_matrix, frame_count
    
    try:
        data = request.get_json()
        frame_data = data['frame'].split(',')[1]
        
        # Decode base64 image
        frame_bytes = base64.b64decode(frame_data)
        frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
        
        # Process frame with YOLO
        results = model(frame, conf=0.25)[0]
        
        # Convert detections to list format
        detections = []
        for box in results.boxes.cpu().numpy():
            detection = {
                'bbox': box.xyxy[0].tolist(),
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0])
            }
            detections.append(detection)
        
        # Update thermal matrix
        frame_count += 1
        heat_matrix = create_thermal_matrix(detections, 
                                          image_size=(frame.shape[1], frame.shape[0]))
        cumulative_heat_matrix += heat_matrix
        
        # Log to TensorBoard every 30 frames
        if frame_count % 30 == 0:
            avg_heat_matrix = cumulative_heat_matrix / frame_count
            fig = plot_thermal_matrix(avg_heat_matrix)
            writer.add_figure('Thermal Matrix/Live', fig, frame_count)
            plt.close(fig)
        
        return jsonify({
            'success': True,
            'detections': detections
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print(f"TensorBoard log directory: {log_dir}")
    print("To view results, run:")
    print(f"tensorboard --logdir={log_dir}")
    app.run(debug=True)