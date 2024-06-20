from ultralytics import YOLO
import torch

# Check for CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the YOLOv8 model
model = YOLO('yolov9e.pt')  # Use the YOLOv8 model variant you prefer

# Set the training parameters
data_config = './data/single_class.yaml'
epochs = 1000
batch_size = 4
image_size = 640
num_workers = 16
# Define the augmentation parameters
augmentation_params = {
    'flipud': 0.2,    # Vertical flip probability
    'fliplr': 0.2,    # Horizontal flip probability
    'mosaic': 0.0,    # Mosaic augmentation probability
    'mixup': 0.0,     # MixUp augmentation probability
    'hsv_h': 0.015,   # HSV-Hue augmentation
    'hsv_s': 0.2,     # HSV-Saturation augmentation
    'hsv_v': 0.2,     # HSV-Value augmentation
    'degrees': 0.2,   # Image rotation degrees
    'translate': 0.2, # Image translation
    'scale': 0.2,     # Image scaling
    'shear': 0.0,     # Image shear
    'perspective': 0.0 # Image perspective
}

# Train the model with augmentations on multiple GPUs
model.train(data=data_config, epochs=epochs, optimizer="Adam",workers=num_workers,batch=batch_size, imgsz=image_size,
            augment=True, device="0",verbose=True, single_cls=True,cos_lr=True,amp=False,plots=True, save=True,**augmentation_params)
