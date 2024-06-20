import shutil

import pandas as pd
import os
import cv2
import numpy as np

train = pd.read_csv("/workspace/xray/train.csv")
test = pd.read_csv("/workspace/xray/test.csv")

labels = ['Atelectasis',
 'Cardiomegaly',
 'Effusion',
 'Infiltrate',
 'Mass',
 'Nodule',
 'Pneumonia',
 'Pneumothorax']


base_folder = '/workspace/xray/yolo_dataset/single_class'
image_folder = '/workspace/xray/images'

def convert_to_yolo(image_width, image_height, top_left_x, top_left_y, width, height):
    """
    Convert bounding box from (top_left_x, top_left_y, width, height) format to YOLO format.

    Parameters:
    image_width (int): Width of the image.
    image_height (int): Height of the image.
    top_left_x (int): X coordinate of the top-left corner of the bounding box.
    top_left_y (int): Y coordinate of the top-left corner of the bounding box.
    width (int): Width of the bounding box.
    height (int): Height of the bounding box.

    Returns:
    tuple: Bounding box in YOLO format (x_center, y_center, width, height) normalized.
    """
    # Calculate the center coordinates
    x_center = top_left_x + width / 2.0
    y_center = top_left_y + height / 2.0

    # Normalize the coordinates by the dimensions of the image
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return x_center, y_center, width, height


def preprocess(df, output_folder):
    for idx, row in df.iterrows():
        filename = row.Image
        label = row.Label

        top_left_x = row.x
        top_left_y = row.y
        width = row.w
        height = row.h

        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        h,w,_ = image.shape

        xc,yc,w,h = convert_to_yolo(w,h,top_left_x, top_left_y, width, height)

        label_id = labels.index(label)

        dest_image_path = os.path.join(output_folder, "images", filename)
        dest_label_path = os.path.join(output_folder, "labels", filename.replace(".png", ".txt"))

        shutil.copy(image_path, dest_image_path)
        with open(dest_label_path, "w") as f:
            f.write(f"0 {xc} {yc} {w} {h}")

        print(idx)



if __name__ == '__main__':
    preprocess(train, os.path.join(base_folder, "train"))
    preprocess(test, os.path.join(base_folder, "val"))