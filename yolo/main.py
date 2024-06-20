from ultralytics import YOLO
import torch

# Check for CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')  # Use the YOLOv8 model variant you prefer

# Set the training parameters
data_config = '/tmp/pycharm_project_808/yolo/data/single_class.yaml'
epochs = 300
batch_size = 32
image_size = 640
num_workers = 64
# Define the augmentation parameters
augmentation_params = {
    'flipud': 0.2,    # Vertical flip probability
    'fliplr': 0.2,    # Horizontal flip probability
    'mosaic': 0.1,    # Mosaic augmentation probability
    'mixup': 0.2,     # MixUp augmentation probability
    'hsv_h': 0.015,   # HSV-Hue augmentation
    'hsv_s': 0.2,     # HSV-Saturation augmentation
    'hsv_v': 0.2,     # HSV-Value augmentation
    'degrees': 0.2,   # Image rotation degrees
    'translate': 0.2, # Image translation
    'scale': 0.2,     # Image scaling
    'shear': 0.2,     # Image shear
    'perspective': 0.2 # Image perspective
}

# Train the model with augmentations on multiple GPUs
model.train(data=data_config, epochs=epochs, workers=num_workers,batch=batch_size, imgsz=image_size, augment=True, device="0,1", **augmentation_params)
