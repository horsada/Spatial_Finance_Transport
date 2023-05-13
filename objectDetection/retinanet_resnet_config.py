# the new config inherits the base configs to highlight the necessary modification
_base_ = '/content/mmdetection/configs/retinanet/retinanet_r101_fpn_1x_coco.py'

fp16 = dict(loss_scale=512.)

# set grad_norm for stability during mixed-precision training
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
    
# optimizer
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9))

# 1. dataset settings
dataset_type = 'CocoDataset'

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

data_root='/content/xView/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True), # xView img shape
    dict(type='RandomCrop', crop_size=(512,512)),
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
val_evaluator = dict(ann_file=data_root + 'val.json')
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

    
model = dict(
    bbox_head=dict(
        num_classes=len(classes),
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128])
        ),
    train_cfg=dict(
        assigner=dict(
            gpu_assign_thr=1
            )
        ),
    backbone=dict(
        with_cp=True)
    )
    
evaluation = dict(interval=1, metric='mAP')
    
workflow = [('train', 1), ('val', 1)]
    
# train, val, test loop config
#train_cfg = dict(max_epochs=10, val_interval=1)
#val_cfg = dict(type='ValLoop') # The validation loop type
#test_cfg = dict(type='TestLoop') # The testing loop type

#log_level = 'INFO'

# We can use the pre-trained model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_mstrain_3x_coco/retinanet_r101_fpn_mstrain_3x_coco_20210720_214650-7ee888e0.pth'