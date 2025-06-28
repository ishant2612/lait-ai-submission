import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import torch
from tracker import PlayerTracker


MODEL_PATH = Path("models/best.pt") 
INPUT_VIDEO = Path("input_video/15sec_input_720p.mp4")
OUTPUT_VIDEO = Path("output_video")
OUTPUT_VIDEO_NAME = "tracked_video.mp4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def run_tracking():

    try:
        model = YOLO(MODEL_PATH)
        print(f"YOLO model loaded from {MODEL_PATH} and will attempt to use {DEVICE}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Please ensure the model file exists at the specified path and PyTorch/CUDA are configured.")
        return

    tracker = PlayerTracker(max_age=90, n_init=3)


    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        print(f"Error: Could not open video file {INPUT_VIDEO}")
        return


    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_path = OUTPUT_VIDEO / OUTPUT_VIDEO_NAME
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    print(f"Output video will be saved to {out_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
 
        results = model(frame, stream=True, device=DEVICE)

        detections = []
        for r in results:
            boxes_data = r.boxes.data
            names = r.names

            for detection_tensor in boxes_data:
                x1, y1, x2, y2, conf, cls = detection_tensor.tolist()

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = float(conf)
                class_id = int(cls)
                class_name = names[class_id]

                # if class_name == 'player' and confidence > 0.5: 
                #     detections.append(([x1, y1, x2, y2], confidence, class_name))
                if class_name == 'player' and confidence > 0.8:
                    # **FIX: Clamp coordinates to frame dimensions**
                    x1_c = max(0, int(x1))
                    y1_c = max(0, int(y1))
                    x2_c = min(width - 1, int(x2))
                    y2_c = min(height - 1, int(y2))
                    
                    # Ensure the box has a valid area after clamping
                    if x2_c > x1_c and y2_c > y1_c:
                        # Append in the correct [x1, y1, x2, y2] format
                        detections.append(([x1_c, y1_c, x2_c, y2_c], confidence, class_name))
        

        print(f"  Frame {frame_count}: Number of YOLO player detections: {len(detections)}")

     
        deepsort_detections = []
        for (x1, y1, x2, y2), conf, class_name in detections:
            w = x2 - x1
            h = y2 - y1
            deepsort_detections.append(([x1, y1, w, h], conf, class_name))

            print(f"DeepSORT Input Bbox (xywh): {[x1, y1, w, h]}, Conf: {conf:.2f}, Class: {class_name}")
            if w <= 5 or h <= 5: 
                print(f"WARNING: Very small detection: w={w}, h={h} for Frame {frame_count}")



        if not isinstance(frame, np.ndarray) or frame.size == 0:
            print(f"WARNING: Frame {frame_count} is invalid or empty. Skipping tracking for this frame.")
            continue

        # print(f"Frame {frame_count}: Frame shape: {frame.shape}, dtype: {frame.dtype}, min/max pixel values: {frame.min()}/{frame.max()}")


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        # frame_normalized = rgb_frame.astype(np.float32) / 255.0
        tracks = tracker.update_tracker(deepsort_detections, frame=rgb_frame) 


        print(f"  Frame {frame_count}: Number of DeepSORT confirmed tracks: {len(tracks)}")


        for track in tracks:
            track_id = track["id"]
            ltrb = track["bbox"]
            class_name = track["class"]

            x1, y1, x2, y2 = map(int, ltrb)
     

            color = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{class_name}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Tracking process complete and video saved.")

if __name__ == "__main__":
    print("Starting detection...")
    run_tracking()
    print("Detection completed.")