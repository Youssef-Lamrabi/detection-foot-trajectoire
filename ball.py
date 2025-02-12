from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os

def process_video(input_path, output_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Make sure this is your trained model path
    
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Properties:")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Total Frames: {total_frames}")
    
    # Initialize video writer
    try:
        if os.name == 'nt':  # For Windows
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:  # For Linux/Mac
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback codec
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Lists to store positions
    ball_positions = []
    
    # Colors
    BALL_COLOR = (0, 255, 0)  # Green for ball
    PLAYER_COLOR = (255, 0, 0)  # Blue for players
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end='\r')
        
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=0.15)  # Lower confidence threshold for ball detection
        
        # Process detections
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                if class_id == 32:  # Sports ball class in COCO dataset
                    # Track ball
                    ball_positions.append((center_x, center_y))
                    if len(ball_positions) > 30:  # Keep only last 30 positions
                        ball_positions.pop(0)
                    
                    # Draw ball detection
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"Ball {confidence:.2f}", (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 2)
                
                elif class_id == 0:  # Person class
                    # Draw player detection (only bounding box, no trajectories)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), PLAYER_COLOR, 2)
                    cv2.putText(frame, f"Player {confidence:.2f}", (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, PLAYER_COLOR, 2)
        
        # Draw ball trajectory with fading effect
        if len(ball_positions) >= 2:
            for i in range(1, len(ball_positions)):
                alpha = i / len(ball_positions)
                color = (
                    int(255 * (1-alpha)),  # Blue
                    int(255),              # Green
                    0                      # Red
                )
                cv2.line(frame, ball_positions[i-1], ball_positions[i], color, 2)
        
        # Add timestamp and info
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"User: azizlg", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write the frame
        out.write(frame)
    
    # Release everything
    cap.release()
    out.release()
    print("\nVideo processing completed!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Input and output paths
    input_video = "video.mp4"  # Your input video
    output_video = "output_video.mp4"  # The processed output video
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Error: Input video file '{input_video}' not found!")
    else:
        print("Starting video processing...")
        process_video(input_video, output_video)