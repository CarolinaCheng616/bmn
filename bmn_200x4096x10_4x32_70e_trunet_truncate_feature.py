# model settings
model = dict(
    type='BMN',
    temporal_dim=200,
    boundary_ratio=0.5,
    num_samples=32,
    num_samples_per_bin=3,
    feat_dim=4096,
    soft_nms_alpha=0.4,
    soft_nms_low_threshold=0.5,
    soft_nms_high_threshold=0.9,
    post_process_top_k=100)
# model training and testing settings
train_cfg = None
test_cfg = dict(average_clips='score')
# dataset settings
dataset_type = 'TruNetDataset'
data_root = 'data/train_mean_2000_10/'
data_root_val = 'data/val_mean_2000_10/'
ann_file_train = 'data/train_meta_10.json'
ann_file_val = 'data/val_meta_10.json'
ann_file_test = 'data/val_meta_10.json'

train_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
val_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second', 'annotations']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
test_pipeline = [
    dict(type='LoadLocalizationFeature'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=['video_name', 'duration_second', 'annotations']),
    dict(type='ToTensor', keys=['raw_feature'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=8,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline,
        data_prefix=data_root_val),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline,
        data_prefix=data_root))

# optimizer
optimizer = dict(
    type='SGD', lr=0.001 / 2, momentum=0.9,
    weight_decay=0.0005)  # this lr is used for 32 gpus, batch_size 128
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[80])
total_epochs = 70
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metrics=['AR@AN'], key_indicator='AR@AN')
log_config = dict(
    interval=5,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/bmn_200x4096x10_4x32_70e_trunet_truncate_feature'
load_from = None
resume_from = None
workflow = [('train', 1)]
output_config = dict(out=f'{work_dir}/results.json', output_format='json')
eval_config = dict(metrics='AR@AN')
