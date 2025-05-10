# Preview
<img src="./smoking_detection_output-1.gif" style="width:400px; height:auto;" />

only detects smoking when cigarate touchs the mouth
<img src="./smoking_detection_output-2.gif" style="width:400px; height:auto;" />

# Smoking Detection

Detect smoking behaviour in images and videos using deep learning and computer vision. This project leverages a custom-trained YOLO (You Only Look Once) object detection model to identify instances of smoking in real-time or from recorded media.

## Features

- **Custom-trained YOLO model** for accurate smoking detection
- Supports both image and video input
- Easy-to-use inference script
- Example media for quick testing

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [Running Inference](#running-inference)
- [Project Structure](#project-structure)
- [Sample Results](#sample-results)
- [Contributing](#contributing)
---

## Installation

1. **Clone the repository:**
```

git clone https://github.com/yashsuman15/Smoking-Detection.git
cd Smoking-Detection

```

2. **Install dependencies:**
- Python 3.7+
- PyTorch
- OpenCV
- YOLOv5 requirements (if retraining)

Install required Python packages:
```

pip install -r requirements.txt

```
*(If `requirements.txt` is missing, install manually: `pip install torch torchvision opencv-python ultralytics`)*

---

## Usage

### Model Training

To train or retrain the model on your own dataset:

1. Prepare your dataset with annotated images (YOLO format).
2. Open and run the Jupyter notebook:
```

YOLO11_Custom_training_for_Object_Detection.ipynb

```
3. The notebook guides you through data loading, model configuration, training, and evaluation.
4. The trained weights will be saved as `best2.pt`.

### Running Inference

Detect smoking in images or videos using the pre-trained model:

```

python smoking_detection2.py --source <path_to_image_or_video> --weights best2.pt

```

- `<path_to_image_or_video>`: Path to your input file (e.g., `media/samples/test1.jpg`)
- The script will display the results with detected smoking instances highlighted.

---

## Project Structure

```

.
├── YOLO11_Custom_training_for_Object_Detection.ipynb  \# Training notebook
├── best2.pt                                          \# Trained model weights
├── smoking_detection2.py                             \# Inference script
├── media/
│   └── samples/                                      \# Sample images/videos
└── README.md                                         \# Project documentation

```

---

## Sample Results


---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the model, add features, or fix bugs.

---

