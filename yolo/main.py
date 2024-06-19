from ultralytics import YOLO
import torch

# Check for CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the YOLOv8 model variant you prefer

# Set the training parameters
data_config = '/tmp/pycharm_project_808/yolo/data/multi_class.yaml'
epochs = 50
batch_size = 128
image_size = 1024
num_workers = 64
# Define the augmentation parameters
augmentation_params = {
    'flipud': 0.5,    # Vertical flip probability
    'fliplr': 0.5,    # Horizontal flip probability
    'mosaic': 1.0,    # Mosaic augmentation probability
    'mixup': 0.5,     # MixUp augmentation probability
    'hsv_h': 0.015,   # HSV-Hue augmentation
    'hsv_s': 0.7,     # HSV-Saturation augmentation
    'hsv_v': 0.4,     # HSV-Value augmentation
    'degrees': 0.0,   # Image rotation degrees
    'translate': 0.1, # Image translation
    'scale': 0.5,     # Image scaling
    'shear': 0.0,     # Image shear
    'perspective': 0.0 # Image perspective
}

# Train the model with augmentations on multiple GPUs
model.train(data=data_config, epochs=epochs, workers=num_workers,batch=batch_size, imgsz=image_size, augment=True, device="0,1", **augmentation_params)
