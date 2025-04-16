# Odyssey-ASL: Real-Time American Sign Language Letter and Word Detection

## Project Overview
**Odyssey-ASL** is a machine learning project designed to bridge communication between American Sign Language (ASL) users and non-signers. The goal was to build a real-time system that can recognize and translate ASL hand signs into letters, words, and potentially full sentences using computer vision and deep learning techniques.

Led by a student engineering team alongside Cal Poly Pomona's Software Engineering Association, this project aimed to provide both an accessible tool for communication and a hands-on learning platform for developing skills in machine learning, computer vision, and agile software practices.

---

## Features
- Video-Based Gesture Recognition: Supports real-time ASL detection from live video feeds.
- Hybrid Model Architecture: Combines Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory Networks (LSTMs) for temporal sequence learning.
- Data Preprocessing Pipeline: Includes frame extraction, resizing, and normalization for consistent input to the model.
- Training Evaluation: Supports model accuracy reporting, loss tracking, and visualization of confusion matrices.

---

## Learning Objectives
This project was designed as a deep-dive into:
- Machine Learning fundamentals: preprocessing, model training, evaluation, and tuning.
- Computer Vision techniques for gesture recognition.
- Model fusion strategies for enhanced performance.
- Agile team collaboration with Scrum-based sprints and code reviews.
- Real-time application development with Python, TensorFlow, and OpenCV.

---

## Dataset
- Letter Detection: [Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- Word Detection: [World-Level American Sign Language (WLASL)](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data)

### Preprocessing Steps:
- Extracted a fixed number of frames per video.
- Resized frames to uniform dimensions.
- Normalized pixel values from `[0, 255]` to `[0, 1]` for neural network compatibility.
- Augmented the dataset using techniques like flipping, rotation, and brightness scaling to improve generalization.

---

## Model Architecture
- 3D Convolutional Neural Networks (3D CNN): For learning spatiotemporal features from video data.
- CNN-LSTM Combination: A hybrid architecture combining CNNs for frame-level spatial analysis and LSTMs for sequence modeling, enabling effective gesture interpretation over time.

---

## Real-Time Implementation
- Framework: TensorFlow / Keras for model training.
- Video Input: OpenCV used to capture real-time webcam streams.
- Output: Model predictions overlayed directly on the video stream for real-time feedback.

---

## Challenges & Solutions
| Challenge | Solution |
|-----------|----------|
| Limited labeled data for ASL gestures | Used data augmentation to expand training data. |
| Model overfitting to training conditions | Standardized background, clothing, and signing speed in both training and testing environments. |
| Real-time inference vs model complexity | Balanced performance using batch normalization, LSTM unit tuning, and convolution layer reduction. |

---

## Installation
1. Create and activate a virtual environment (recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

2. Install project dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and place the dataset folders (`grouped_videos_10`) in the Words->Data directory.  

---

## Project Timeline
| Sprint | Focus |
|--------|-------|
| Sprint 1 | Project kickoff, research, and planning |
| Sprint 2 | Dataset acquisition and preprocessing |
| Sprint 3 | Model selection and baseline implementation |
| Sprint 4 | Model tuning and feature engineering |
| Sprint 5 | Real-time testing with MediaPipe integration |
| Sprint 6 | Advanced techniques and model fusion (if applicable) |
| Sprint 7 | Final prototype and demonstration |

---

## Contributors
- Project Lead: Tony Gonzalez  
- Co-Author: Prerna Joshi  
- Nhan Thai
- Iker Goni
- Kayla Scarberry
- Michael Castillo 
- Brisa Ramirez
- Kevin Kopcinski
- Vincent Terrelonge
- Ben Stevenson
- Kathee Avendano

---

## References
- [CPP Software Engineering Association](https://cppsea.com/)
- [Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- [WLASL Dataset](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data)

---