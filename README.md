# YOLOv8 Player Tracking with DeepSORT Re-Identification

This project demonstrates real-time player tracking in a video feed using the YOLOv8 object detection model and the DeepSORT algorithm for persistent object tracking and re-identification.


## Table of Contents

  * [Objective]

  * [Key Technologies]

  * [Setup and Installation]

  * [How to Run]

  * [Project Structure]

  * [Development Approach]
## Objective

Given a 15-second video clip, the goal is to identify each player and ensure that players who go out of frame and reappear are assigned the same identity as before.

### Key Requirements:

  * Use an object detection model to detect players throughout the clip.

  * Assign a unique and persistent ID to each player.

  * Maintain the same ID for a player even after they re-enter the frame.

  * The solution should simulate a real-time re-identification and player tracking system.

## Key Technologies

  * **Python 3.10**

  * **YOLOv8:** For high-performance, real-time object detection.

  * **DeepSORT (with TorchReID):** For state-of-the-art object tracking. DeepSORT uses a Kalman filter to predict object motion and a deep learning-based Re-ID model (OSNet) to handle identity re-assignment after occlusions.

  * **PyTorch:** The underlying deep learning framework for both YOLO and TorchReID.

  * **OpenCV:** For video processing (reading, writing, and displaying frames).

  * **NumPy:** For numerical operations.

## Setup and Installation

Follow these steps carefully to set up the project environment. The installation is complex due to the dependencies of the tracking library.

### 1\. Prerequisites

  * **Python 3.10** or higher.

  * **Git** for cloning the repository.

  * **Microsoft C++ Build Tools:** This is **required** to compile parts of the `torchreid` library.

      * Go to the [Visual C++ Build Tools website](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

      * Download the installer.

      * During installation, select the **"Desktop development with C++"** workload and click "Install".

      * **Restart** your **system** after the installation is complete.

### 2\. Clone the Repository

```
git clone https://github.com/ishant2612/lait_ai_submission.git
cd lait-ai-submission

```


### 3\. Install Dependencies

It is highly recommended to use a virtual environment.

```
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

```

### 4\. Install `torchreid` from Source (Crucial Step)

The standard `pip install torchreid` package is incorrect. You must install it from the official source.

```
# 1. Clone the official repository
git clone https://github.com/KaiyangZhou/deep-person-reid.git

# 2. Navigate into the cloned directory
cd deep-person-reid

# 3. Install the library in development mode
# This step requires the C++ Build Tools to be installed first.
python setup.py develop

```

### 5\. Download Models

You need two model files. Place them inside the `models/` directory.

  * **YOLOv8 Detection Model:** `best.pt`

      * This project requires a YOLOv8 model trained to detect players.

      * You must provide your own trained model file and name it `best.pt`.

      * Place this file in the `models/` directory.

      * **Note:** Due to its large size, this file should **not** be committed to GitHub. Anyone cloning the repository must add it manually.

  * **Re-Identification Model (OSNet):** `osnet_x1_0_imagenet.pth`

      * Download this pre-trained model from this link: [**osnet\_x1\_0\_imagenet.pth**](https://www.google.com/search?q=https://drive.google.com/uc%3Fid%3D1V77c22t1Gztc6fPSiR3D20yqAg5qTA6n)

      * Place the downloaded file at `models/osnet_x1_0_imagenet.pth`.

## How to Run

After completing the setup, run the main script from the project's root directory:

```
python scripts/main.py

```

A window will appear showing the real-time tracking. The final processed video will be saved in the `output_video/` directory. Press 'q' to quit the live preview.

## Project Structure

```
├── models/
│   ├── best.pt               # YOLOv8 detection model (must be provided manually)
│   └── osnet_x1_0_imagenet.pth # TorchReID feature extractor model
├── input_video/
│   └── 15sec_input_720p.mp4  # Sample input video
├── output_video/
│   └── tracked_video.mp4     # Generated output video
├── scripts/
│   ├── main.py               # Main script to run the tracking
│   └── tracker.py            # Wrapper class for the DeepSORT tracker
├── .gitignore                # Specifies files for Git to ignore
└── README.md                 # This file

```

## Development Approach

The implementation followed a systematic process to integrate object detection with multi-object tracking:

1.  **Dependency Management:** The `torchreid` library is a core component for the tracker's re-identification capabilities. We determined that the standard PyPI package was insufficient for our needs.

      * **Approach:** We installed the library directly from the official GitHub source. This required compiling C++ extensions, for which we first set up the "Microsoft C++ Build Tools" environment.

2.  **Data Flow Pipeline:** A robust pipeline was established to pass information from the detector to the tracker.

      * **Approach:** The raw output from the YOLOv8 model (bounding boxes, confidence scores, and class names) was carefully formatted into the specific `[([x1, y1, x2, y2], confidence, class)]` structure required by the DeepSORT tracker. This ensures the tracker receives clean, correctly formatted data for every frame.

3.  **Tracker Configuration:** To ensure reliable feature extraction for re-identification, the tracker was explicitly configured.

      * **Approach:** We configured the `DeepSort` object to use a specific, high-performance Re-ID model (OSNet). By providing the model architecture name and the direct path to its pre-trained weights file, we guarantee that the tracker uses a consistent and powerful feature extractor. This is crucial for accurately maintaining player IDs across occlusions and re-appearances.
