import pandas as pd
import numpy as np
import cv2
import os
from glob import glob
import shutil
from mmengine.fileio import dump, load

multiple_labels = ['Atelectasis',
                   'Cardiomegaly',
                   'Effusion',
                   'Infiltrate',
                   'Mass',
                   'Nodule',
                   'Pneumonia',
                   'Pneumothorax']

multiple_categories = [{"id": multiple_labels.index(x), "name": x} for x in multiple_labels]

single_label = ['Disease']
single_category = [{"id": single_label.index(x), "name": x} for x in single_label]


def convert_bbox(bbox):
    x_c, y_c, w, h = bbox
    x_top_left = x_c - w / 2
    y_top_left = y_c - h / 2
    return x_top_left, y_top_left, w, h


def process(folder, output_annotation, output_image_folder, is_multiple=True):
    items = glob(os.path.join(folder, "images", "*.png")) + glob(os.path.join(folder, "images", "*.jpg")) + glob(
        os.path.join(folder, "images", "*.jpeg"))
    annotations = []
    images = []

    annotation_id = 0
    print("started...")
    for idx, file_path in enumerate(items):
        filename = os.path.basename(file_path)

        dst_image_path = os.path.join(output_image_folder, filename)
        shutil.copy(file_path, dst_image_path)

        image = cv2.imread(file_path)
        image_h, image_w, _ = image.shape
        image_data = dict(id=idx, file_name=filename, height=image_h, width=image_w)
        images.append(image_data)
        extension = filename.split(".")[-1]
        label_path = file_path.replace('/images/', '/labels/').replace(f".{extension}", '.txt')
        with open(label_path, 'r') as f:
            label_content = np.array([x.split() for x in f.read().strip().split("\n")]).astype(np.float32).reshape(-1,
                                                                                                                   5)

        for ann in label_content:
            if is_multiple:
                category_id = ann[0]
            else:
                category_id = 0
            bbox = ann[1:]
            bbox[0::2] *= image_w
            bbox[1::2] *= image_h
            x_top_left, y_top_left, w, h = convert_bbox(bbox)

            anno_data = dict(id=annotation_id, image_id=idx, category_id=int(category_id),
                             bbox=[x_top_left, y_top_left, w, h], area=w * h, iscrowd=0)
            annotations.append(anno_data)
            print(f"{idx} {annotation_id} {filename}")
            annotation_id += 1
    if is_multiple:
        categories_data = multiple_categories
    else:
        categories_data = single_category

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories_data)
    dump(coco_format_json, output_annotation)


if __name__ == '__main__':
    process("/data/yolo_dataset/tmp", "/data/single/train.json", "/data/single", is_multiple=False)
    process("/data/yolo_dataset/val", "/data/single/test.json", "/data/single", is_multiple=False)
