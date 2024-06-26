/opt/conda/bin/python3 train.py --name xray --project /data/yolo_dataset/yolov5_output_dir \
  --data /tmp/pycharm_project_791/yolo/configs/xray.yaml --imgsz 1024 --batch-size 12 \
  --device 0 --optimizer Adam --seed 42 --weights /data/yolo_dataset/weights/yolov5x.pt --cache --evolve