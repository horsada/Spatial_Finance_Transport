# the new config inherits the base configs to highlight the necessary modification
_base_ = [
    '/content/mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '/content/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/content/mmdetection/configs/_base_/schedules/schedule_1x.py', '/content/mmdetection/configs/_base_/default_runtime.py'
    ]
    
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

classes = ('Fixed-wing Aircraft', 
'Small Aircraft',
'Cargo Plane',
'Helicopter',
'Passenger Vehicle',
'Small Car',
'Bus',
'Pickup Truck',
'Utility Truck',
'Truck',
'Cargo Truck',
'Truck w/Box',
'Truck Tractor',
'Trailer',
'Truck w/Flatbed',
'Truck w/Liquid',
'Crane Truck',
'Railway Vehicle',
'Passenger Car',
'Cargo Car',
'Flat Car',
'Tank car',
'Locomotive',
'Maritime Vessel',
'Motorboat',
'Sailboat',
'Tugboat',
'Barge',
'Fishing Vessel',
'Ferry',
'Yacht',
'Container Ship',
'Oil Tanker',
'Engineering Vehicle',
'Tower crane',
'Container Crane',
'Reach Stacker',
'Straddle Carrier',
'Mobile Crane',
'Dump Truck',
'Haul Truck',
'Scraper/Tractor',
'Front loader/Bulldozer',
'Excavator',
'Cement Mixer',
'Ground Grader',
'Hut/Tent',
'Shed',
'Building',
'Aircraft Hangar',
'Damaged Building',
'Facility',
'Construction Site',
'Vehicle Lot',
'Helipad',
'Storage Tank',
'Shipping container lot',
'Shipping Container',
'Pylon',
'Tower'
)

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5),
    bbox_head=dict(
        num_classes=len(classes)
        ),
    train_cfg=dict(
        assigner=dict(
        gpu_assign_thr=1
            )
        )
    )

# optimizer
#optimizer=dict(type='AdamW')
#optimizer=dict(type='AdamW', lr=0.0003, weight_decay=0.0001)
#optim_wrapper = dict(optimizer=dict(type='Adam'))
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.001))


fp16 = dict(loss_scale=512.)

# set grad_norm for stability during mixed-precision training
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))


# 1. dataset settings
dataset_type = 'CocoDataset'

data_root='/content/xView/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train_images/'),
        pipeline=train_pipeline
        )
    )

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='train_images/')
        )
    )

test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(
    ann_file=data_root+'train.json',
    type='CocoMetric',
    metric=['bbox']
    )
test_evaluator = val_evaluator

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=5),
    visualization=dict(type='DetVisualizationHook', draw=True)
    )
    
vis_backends = [dict(type='LocalVisBackend'),
            dict(type='WandbVisBackend')]

visualizer = dict(
    type='DetLocalVisualizer', 
    vis_backends=vis_backends, 
    name='visualizer'
    )
    
# train, val, test loop config
#train_cfg = dict(max_epochs=10, val_interval=1)
#val_cfg = dict(type='ValLoop') # The validation loop type
#test_cfg = dict(type='TestLoop') # The testing loop type

#log_level = 'INFO'