import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import cv2
import json

# Define the augmentation pipeline
augmentation_pipeline = A.Compose([
    A.VerticalFlip(p=0.2),
    A.HorizontalFlip(p=0.2),
    A.RandomRotate90(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.Blur(p=0.2),
    A.GaussNoise(p=0.2),
    A.Resize(1024, 1024)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def load_image_and_labels(image_path, label_path):
    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))

    # Load labels
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))

    return image, bboxes, class_labels


def save_augmented_data(augmented_image, augmented_bboxes, augmented_labels, output_image_path, output_label_path):
    # Save the augmented image
    augmented_image_pil = Image.fromarray(augmented_image)
    augmented_image_pil.save(output_image_path)

    # Save the augmented labels
    with open(output_label_path, 'w') as file:
        for bbox, label in zip(augmented_bboxes, augmented_labels):
            bbox_str = ' '.join(map(str, bbox))
            file.write(f"{label} {bbox_str}\n")


def augment_and_save(image_path, label_path, output_image_dir, output_label_dir, num_augmentations=5):
    image, bboxes, class_labels = load_image_and_labels(image_path, label_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    for i in range(num_augmentations):
        augmented = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_labels = augmented['class_labels']

        output_image_path = os.path.join(output_image_dir, f"{name}_aug_{i}{ext}")
        output_label_path = os.path.join(output_label_dir, f"{name}_aug_{i}.txt")

        save_augmented_data(augmented_image, augmented_bboxes, augmented_labels, output_image_path, output_label_path)


# Paths
train_images_dir = "/data/yolo_dataset/train/images"
train_labels_dir = "/data/yolo_dataset/train/labels"
augmented_images_dir = "/data/yolo_dataset/augmented/images"
augmented_labels_dir = "/data/yolo_dataset/augmented/labels"
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)

index = 0
# Augment and save each image and its corresponding labels
for image_file in os.listdir(train_images_dir):
    if image_file.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(train_images_dir, image_file)
        label_path = os.path.join(train_labels_dir,
                                  image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
        augment_and_save(image_path, label_path, augmented_images_dir, augmented_labels_dir, num_augmentations=5)
        print(image_path, index)
        index+=1