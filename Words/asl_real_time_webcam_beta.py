import os
import datetime as dt
import torch
import cv2
import numpy as np
from torchvision.models.video import r3d_18
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image  # Import PIL for NumPy to PIL conversion

# Define preprocessing parameters
frame_size = (112, 112)
transform = Compose([
    Resize(frame_size),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

# Function to detect class names dynamically from the root folder
def detect_class_names(root_folder):
    class_names = [
        class_name for class_name in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, class_name))
    ]
    return class_names

# Function to load the trained model
def load_model(model_path, num_classes, device):
    model = r3d_18(weights=None)  # Use weights=None instead of pretrained=True
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path}")
    return model


# Function to capture video frames from the webcam
def capture_video_sequence(cap, num_frames=16):
    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert NumPy array to PIL image
        frame_pil = Image.fromarray(frame_rgb)
        # Apply preprocessing
        frame_transformed = transform(frame_pil)
        frames.append(frame_transformed)
    return torch.stack(frames, dim=1).unsqueeze(0)  # Shape: (1, C, T, H, W)


# Function to test the model with a real webcam
def test_model_with_webcam(model, class_labels, device, target_fps=60):
    cap = cv2.VideoCapture(0)  # 0 is the default webcam

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Set webcam FPS
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Webcam FPS set to: {actual_fps:.2f}")

    print("Press 'q' to quit.")

    while True:
        # Capture a sequence of frames
        video_tensor = capture_video_sequence(cap)

        if video_tensor.size(2) < 16:  # Check if the sequence is complete
            continue

        # Move tensor to the device
        video_tensor = video_tensor.to(device)

        # Perform prediction
        with torch.no_grad():
            outputs = model(video_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = class_labels[predicted.item()]

        # Display the prediction on the webcam feed
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Overlay prediction text
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


# Main function to run the entire pipeline
def main():
    # root_folder = "preprocessed_tensors_1"  # Update with your dataset's root folder
    # root_folder = "preprocessed_tensors_3"  # Update with your dataset's root folder
    root_folder = "preprocessed_tensors_5"  # Update with your dataset's root folder
    model_path = "3D_CNN_Model_5Classes_2024_12_30__19_24_34_Loss1.1161_Acc75.00.pth"  # Update with your model path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Detect class names
    class_labels = detect_class_names(root_folder)
    print(f"Detected Classes: {class_labels}")
    num_classes = len(class_labels)

    # Load the trained model
    model = load_model(model_path, num_classes, device)

    # Test the model with a real webcam
    test_model_with_webcam(model, class_labels, device)


if __name__ == "__main__":
    main()
