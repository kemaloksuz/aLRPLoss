_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco500_detection_augm.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(type='APLossRetinaHead',
    	anchor_generator=dict(scales_per_octave=2),
    	bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
    	loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# optimizer
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(step=[60, 80])
total_epochs = 100
