from ultralytics import YOLO
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

def create_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = YOLO('yolov8x.pt')  # Use the YOLOv8 model variant you prefer

data_config = './data.yaml'
epochs = 100
batch_size = 18
image_size = 1024
num_workers = 32
augmentation_params = {
    'flipud': 0.1,    # Vertical flip probability
    'fliplr': 0.1,    # Horizontal flip probability
    'mosaic': 0.0,    # Mosaic augmentation probability
    'mixup': 0.0,     # MixUp augmentation probability
    'hsv_h': 0.015,   # HSV-Hue augmentation
    'hsv_s': 0.1,     # HSV-Saturation augmentation
    'hsv_v': 0.1,     # HSV-Value augmentation
    'degrees': 0.1,   # Image rotation degrees
    'translate': 0.1, # Image translation
    'scale': 0.1,     # Image scaling
    'shear': 0.0,     # Image shear
    'perspective': 0.0 # Image perspective
}

model.train(data=data_config, epochs=epochs,workers=num_workers,batch=batch_size, imgsz=image_size,
            single_cls=True,save_period=10,augment=True, device="0,1",optimizer="AdamW",
            project="/data/new_dataset/output_dir",name="final_xray",verbose=True,plots=True, save=True,
            lr0=0.01,
            lrf=0.001,
            scheduler=create_scheduler,**augmentation_params)
