import torch
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the YOLOv8 model
model = YOLO('yolov8x.pt')  # Use the YOLOv8 model variant you prefer

# Set the training parameters
data_config = './configs/xray.yaml'
epochs = 1000
batch_size = 10
image_size = 1024
num_workers = 16
patience = 0
# Define the augmentation parameters
augmentation_params = {'flipud': 0.2,  # Vertical flip probability
    'fliplr': 0.2,  # Horizontal flip probability
    'mosaic': 0.0,  # Mosaic augmentation probability
    'mixup': 0.0,  # MixUp augmentation probability
    'hsv_h': 0.015,  # HSV-Hue augmentation
    'hsv_s': 0.2,  # HSV-Saturation augmentation
    'hsv_v': 0.2,  # HSV-Value augmentation
    'degrees': 0.2,  # Image rotation degrees
    'translate': 0.2,  # Image translation
    'scale': 0.2,  # Image scaling
    'shear': 0.0,  # Image shear
    'perspective': 0.0  # Image perspective
}

# Train the model with augmentations on multiple GPUs
model.train(data=data_config, epochs=epochs, workers=num_workers, batch=batch_size, imgsz=image_size, save_period=10,
            augment=True, device="0", optimizer="AdamW", project="/data/yolo_dataset/output_dir", name="xray",
            verbose=True, plots=True, save=True, patience=patience, **augmentation_params)
