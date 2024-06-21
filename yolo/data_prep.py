import os
import shutil

import cv2
import pandas as pd

train = pd.read_csv("/data/yolo_dataset/train_unique.csv")
test = pd.read_csv("/data/yolo_dataset/test_unique.csv")

labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax']

base_folder = '/data/yolo_dataset'
image_folder = '/data/images'


def convert_to_yolo(image_width, image_height, top_left_x, top_left_y, width, height):
    x_center = top_left_x + width / 2.0
    y_center = top_left_y + height / 2.0

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
        img_h, img_w, _ = image.shape

        xc, yc, w, h = convert_to_yolo(image_width=img_w, image_height=img_h, top_left_x=top_left_x,
                                       top_left_y=top_left_y, width=width, height=height)

        label_id = labels.index(label)

        dest_image_path = os.path.join(output_folder, "images", filename)
        dest_label_path = os.path.join(output_folder, "labels", filename.replace(".png", ".txt"))

        shutil.copy(image_path, dest_image_path)
        with open(dest_label_path, "w") as f:
            f.write(f"{label_id} {xc} {yc} {w} {h}")

        print(idx)


if __name__ == '__main__':
    preprocess(train, os.path.join(base_folder, "train"))
    preprocess(test, os.path.join(base_folder, "val"))
