_base_ = '/mmdetection/configs/detectors/cascade-rcnn_r50-rfp_1x_coco.py'

data_root = '/data/augmented/'  # dataset root
train_batch_size_per_gpu = 8
train_num_workers = 32

max_epochs = 100
stage2_num_epochs = 1
base_lr = 0.00008

metainfo = {
    'classes': (
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltrate',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
    ),
    'palette': [
        (220, 20, 60),
        (200, 201, 60),
        (184, 20, 50),
        (140, 50, 60),
        (23, 20, 60),
        (55, 20, 60),
        (77, 20, 60),
        (10, 20, 60),
    ]
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=''),
        ann_file='train.json'),
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img=''),
        ann_file='test.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'test.json')

test_evaluator = val_evaluator

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

albu_train_transforms = [
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.1,
        rotate_limit=20,
        scale_limit=0.2,
        p=0.9
    ),
    dict(
        type='IAAAffine',
        shear=(-10.0, 10.0),
        p=0.5
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", p=1.0, blur_limit=7),
            dict(type="GaussianBlur", p=1.0, blur_limit=7),
            dict(type="MedianBlur", p=1.0, blur_limit=7),
        ],
        p=0.2,
    ),
    dict(type="RandomBrightnessContrast", p=0.9, brightness_limit=0.25, contrast_limit=0.25),
    dict(
        type='OneOf',
        transforms=[
            dict(type='IAAAdditiveGaussianNoise', scale=(0.01 * 255, 0.05 * 255), p=1.0),
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=1.0)
        ]
    ),
    dict(type='HorizontalFlip', p=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        keymap=dict(img="image", gt_bboxes="bboxes"),
        update_pad_shape=False,
        skip_img_without_anno=True,
        bbox_params=dict(type="BboxParams", format="pascal_voc", label_fields=['gt_labels'], filter_lost_elements=True, min_visibility=0.1, min_area=1),
    ),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

data = dict(
    samples_per_gpu=train_batch_size_per_gpu,
    workers_per_gpu=train_num_workers,
    train=[
        dict(
            type='CocoDataset',
            data_root=data_root,
            metainfo=metainfo,
            ann_file='train.json',
            pipeline=[
                dict(
                    type="Albu",
                    transforms=albu_train_transforms,
                    keymap=dict(img="image", gt_bboxes="bboxes"),
                    update_pad_shape=False,
                    skip_img_without_anno=True,
                    bbox_params=dict(type="BboxParams", format="pascal_voc", label_fields=['gt_labels'], filter_lost_elements=True, min_visibility=0.1, min_area=1),
                ),
                dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
        ),
        dict(
            type='CocoDataset',
            data_root="/data/augmented/",
            metainfo=metainfo,
            ann_file='augmented.json',
            pipeline=[
                dict(
                    type="Albu",
                    transforms=albu_train_transforms,
                    keymap=dict(img="image", gt_bboxes="bboxes"),
                    update_pad_shape=False,
                    skip_img_without_anno=True,
                    bbox_params=dict(type="BboxParams", format="pascal_voc", label_fields=['gt_labels'], filter_lost_elements=True, min_visibility=0.1, min_area=1),
                ),
                dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.0),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            ]
        )
    ],
    val=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        pipeline=test_pipeline),
    test=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        pipeline=test_pipeline)
)

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# load cascade model pre-trained weight
load_from = '/data/checkpoints/cascade_rcnn_r50_rfp_1x_coco-8cf51bfd.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

# Distributed Training Config
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]

# Set random seed to ensure deterministic behavior
seed = 42

work_dir = "/data/work_dir/augmented"