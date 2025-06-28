# from deep_sort_realtime.deepsort_tracker import DeepSort
# import numpy as np
# import torch

# class PlayerTracker:
#     def __init__(self, max_age=90,n_init = 1):

#         self.tracker = DeepSort(max_age=max_age, n_init=n_init, nms_max_overlap=1.0, embedder="mobilenet",bgr=False)

#     def update_tracker(self, detections, frame):
#         formatted_detections = detections

#         tracks = self.tracker.update_tracks(formatted_detections, frame) 

#         confirmed_tracks = []

#         for track in tracks:
   
#             if not track.is_confirmed():
#                 continue

#             track_id = track.track_id
#             ltrb = track.to_ltrb() 
#             detection_class = track.get_det_class()

#             confirmed_tracks.append({
#                 "id": track_id,
#                 "bbox": ltrb,
#                 "class": detection_class
#             })
#         return confirmed_tracks


from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import torch
from pathlib import Path

class PlayerTracker:
    def __init__(self, max_age=90, n_init=3):

        # Path to the pre-trained Re-ID model weights
        model_path = Path("models/osnet_x1_0_imagenet.pth")

        # Check if the model file exists
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Re-ID model file not found at {model_path}. "
                "Please download 'osnet_x1_0_imagenet.pth' and place it in the 'models' directory."
            )

        # Initialize the DeepSort tracker with an explicitly configured Re-ID model
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=0.2,
            embedder="torchreid",
            bgr=False,  
            embedder_gpu=torch.cuda.is_available(),  # Use GPU if available
            embedder_model_name="osnet_x1_0", # The name of the model architecture
            embedder_wts=str(model_path)      # The path to the weights file
        )

    def update_tracker(self, detections, frame):
        # This part of your code remains the same as it is correct
        tracks = self.tracker.update_tracks(detections, frame=frame) 

        confirmed_tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb() 
            detection_class = track.get_det_class()

            confirmed_tracks.append({
                "id": track_id,
                "bbox": ltrb,
                "class": detection_class
            })
        return confirmed_tracks