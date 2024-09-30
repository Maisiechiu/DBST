_base_ = ['../../_base_/default_runtime.py']
# model settings
model = dict(type='Recognizer3D',
             backbone=dict(
                 type='VisionTransformer',
                 img_size=224,
                 patch_size=16,
                 embed_dims=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 num_frames=16,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 drop_path_rate=0.0,
                 drop_rate=0.0,
             ),
             cls_head=dict(type='TimeSformerHead',
                           num_classes=2,
                           in_channels=768,
                           average_clips='prob',
                           loss_cls=dict(type='CBFocalLoss',
                                         samples_per_cls=[5754, 28727])),
             data_preprocessor=dict(type='ActionDataPreprocessor',
                                    mean=[123.675, 116.28, 103.53],
                                    std=[58.395, 57.12, 57.375],
                                    format_shape='NCTHW'))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = ''
data_root_val = ''
data_root_test = '/home/jovyan/dataset/Celeb-DF-v2/'
ann_file_train = '/home/jovyan/temporaldeepfake/mmaction2/data/rawframes2_random_margin_5frames.txt'
ann_file_val = '/home/jovyan/temporaldeepfake/mmaction2/data/rawframes2_cdf_identity_part.txt'
ann_file_test = '/home/jovyan/temporaldeepfake/mmaction2/data/test_list_DFDC_full_with_identity.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DeepfakeSampleFrames',
         clip_len=16,
         frame_interval=1,
         num_clips=1),
    dict(type='DeepfakeFrameDecode2', **file_client_args),
    dict(type='AlbumentationsColorJitter'),
    dict(type='Flip', flip_ratio=0.5),
    dict(input_size=224,
         max_wh_scale_gap=1,
         random_crop=False,
         scales=(
             0.85,
             0.875,
             0.9,
         ),
         type='MultiScaleCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='DeepfakeSampleFrames',
         clip_len=16,
         frame_interval=1,
         num_clips=1),
    dict(type='DeepfakeFrameDecode2', **file_client_args),
    dict(type='Resize', scale=(270, 270)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='DeepfakeSampleFrames',
         clip_len=16,
         frame_interval=1,
         num_clips=1,
         test_mode=True),
    dict(type='DeepfakeFrameDecode2', **file_client_args),
    dict(type='Resize', scale=(270, 270), keep_ratio=False),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(batch_size=3,
                        num_workers=8,
                        persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),
                        dataset=dict(type=dataset_type,
                                     ann_file=ann_file_train,
                                     filename_tmpl='{:03d}.png',
                                     modality='RGB',
                                     data_prefix=dict(img=data_root),
                                     pipeline=train_pipeline))
val_dataloader = dict(batch_size=24,
                      num_workers=12,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      dataset=dict(type=dataset_type,
                                   ann_file=ann_file_val,
                                   filename_tmpl='{:03d}.png',
                                   modality='RGB',
                                   data_prefix=dict(img=data_root_val),
                                   pipeline=val_pipeline,
                                   test_mode=False))
test_dataloader = dict(batch_size=32,
                       num_workers=12,
                       persistent_workers=True,
                       sampler=dict(type='DefaultSampler', shuffle=False),
                       dataset=dict(type=dataset_type,
                                    ann_file=ann_file_test,
                                    filename_tmpl='{:03d}.png',
                                    modality='RGB',
                                    data_prefix=dict(img=data_root_test),
                                    pipeline=test_pipeline,
                                    test_mode=True))
val_evaluator = dict(type='AucMetric')

train_cfg = dict(type='EpochBasedTrainLoop',
                 max_epochs=60,
                 val_begin=1,
                 val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,  # by epoch
        end=5),
    dict(
        T_max=55,  # 增加衰减时间
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=60,
        type='CosineAnnealingLR'),
]
optim_wrapper = dict(optimizer=dict(betas=(0.9, 0.999),
                                    lr=0.00001,
                                    type='AdamW',
                                    weight_decay=0.05),
                     paramwise_cfg=dict(custom_keys=dict({
                         'backbone.blocks.0.attn.proj.bias':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.attn.proj.weight':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.attn.q_bias':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.attn.qkv.weight':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.attn.v_bias':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.mlp.layers.0.0.bias':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.mlp.layers.0.0.weight':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.mlp.layers.1.bias':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.mlp.layers.1.weight':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.norm1.bias':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.norm1.weight':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.norm2.bias':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.0.norm2.weight':
                         dict(lr_mult=0.2824295364810001),
                         'backbone.blocks.1.attn.proj.bias':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.attn.proj.weight':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.attn.q_bias':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.attn.qkv.weight':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.attn.v_bias':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.mlp.layers.0.0.bias':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.mlp.layers.0.0.weight':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.mlp.layers.1.bias':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.mlp.layers.1.weight':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.norm1.bias':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.norm1.weight':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.norm2.bias':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.1.norm2.weight':
                         dict(lr_mult=0.31381059609000006),
                         'backbone.blocks.10.attn.proj.bias':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.attn.proj.weight':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.attn.q_bias':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.attn.qkv.weight':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.attn.v_bias':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.mlp.layers.0.0.bias':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.mlp.layers.0.0.weight':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.mlp.layers.1.bias':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.mlp.layers.1.weight':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.norm1.bias':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.norm1.weight':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.norm2.bias':
                         dict(lr_mult=0.81),
                         'backbone.blocks.10.norm2.weight':
                         dict(lr_mult=0.81),
                         'backbone.blocks.11.attn.proj.bias':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.attn.proj.weight':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.attn.q_bias':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.attn.qkv.weight':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.attn.v_bias':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.mlp.layers.0.0.bias':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.mlp.layers.0.0.weight':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.mlp.layers.1.bias':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.mlp.layers.1.weight':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.norm1.bias':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.norm1.weight':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.norm2.bias':
                         dict(lr_mult=0.9),
                         'backbone.blocks.11.norm2.weight':
                         dict(lr_mult=0.9),
                         'backbone.blocks.2.attn.proj.bias':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.attn.proj.weight':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.attn.q_bias':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.attn.qkv.weight':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.attn.v_bias':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.mlp.layers.0.0.bias':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.mlp.layers.0.0.weight':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.mlp.layers.1.bias':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.mlp.layers.1.weight':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.norm1.bias':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.norm1.weight':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.norm2.bias':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.2.norm2.weight':
                         dict(lr_mult=0.3486784401000001),
                         'backbone.blocks.3.attn.proj.bias':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.attn.proj.weight':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.attn.q_bias':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.attn.qkv.weight':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.attn.v_bias':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.mlp.layers.0.0.bias':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.mlp.layers.0.0.weight':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.mlp.layers.1.bias':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.mlp.layers.1.weight':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.norm1.bias':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.norm1.weight':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.norm2.bias':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.3.norm2.weight':
                         dict(lr_mult=0.3874204890000001),
                         'backbone.blocks.4.attn.proj.bias':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.attn.proj.weight':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.attn.q_bias':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.attn.qkv.weight':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.attn.v_bias':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.mlp.layers.0.0.bias':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.mlp.layers.0.0.weight':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.mlp.layers.1.bias':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.mlp.layers.1.weight':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.norm1.bias':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.norm1.weight':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.norm2.bias':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.4.norm2.weight':
                         dict(lr_mult=0.4304672100000001),
                         'backbone.blocks.5.attn.proj.bias':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.attn.proj.weight':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.attn.q_bias':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.attn.qkv.weight':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.attn.v_bias':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.mlp.layers.0.0.bias':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.mlp.layers.0.0.weight':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.mlp.layers.1.bias':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.mlp.layers.1.weight':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.norm1.bias':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.norm1.weight':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.norm2.bias':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.5.norm2.weight':
                         dict(lr_mult=0.4782969000000001),
                         'backbone.blocks.6.attn.proj.bias':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.attn.proj.weight':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.attn.q_bias':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.attn.qkv.weight':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.attn.v_bias':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.mlp.layers.0.0.bias':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.mlp.layers.0.0.weight':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.mlp.layers.1.bias':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.mlp.layers.1.weight':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.norm1.bias':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.norm1.weight':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.norm2.bias':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.6.norm2.weight':
                         dict(lr_mult=0.531441),
                         'backbone.blocks.7.attn.proj.bias':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.attn.proj.weight':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.attn.q_bias':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.attn.qkv.weight':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.attn.v_bias':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.mlp.layers.0.0.bias':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.mlp.layers.0.0.weight':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.mlp.layers.1.bias':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.mlp.layers.1.weight':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.norm1.bias':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.norm1.weight':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.norm2.bias':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.7.norm2.weight':
                         dict(lr_mult=0.5904900000000001),
                         'backbone.blocks.8.attn.proj.bias':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.attn.proj.weight':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.attn.q_bias':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.attn.qkv.weight':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.attn.v_bias':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.mlp.layers.0.0.bias':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.mlp.layers.0.0.weight':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.mlp.layers.1.bias':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.mlp.layers.1.weight':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.norm1.bias':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.norm1.weight':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.norm2.bias':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.8.norm2.weight':
                         dict(lr_mult=0.6561),
                         'backbone.blocks.9.attn.proj.bias':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.attn.proj.weight':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.attn.q_bias':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.attn.qkv.weight':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.attn.v_bias':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.mlp.layers.0.0.bias':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.mlp.layers.0.0.weight':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.mlp.layers.1.bias':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.mlp.layers.1.weight':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.norm1.bias':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.norm1.weight':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.norm2.bias':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.blocks.9.norm2.weight':
                         dict(lr_mult=0.7290000000000001),
                         'backbone.fc_norm.bias':
                         dict(lr_mult=1.0),
                         'backbone.fc_norm.weight':
                         dict(lr_mult=1.0),
                         'backbone.patch_embed.projection.bias':
                         dict(lr_mult=0.2541865828329001),
                         'backbone.patch_embed.projection.weight':
                         dict(lr_mult=0.2541865828329001),
                         'cls_head.fc_cls.bias':
                         dict(lr_mult=1.0),
                         'cls_head.fc_cls.weight':
                         dict(lr_mult=1.0)
                     })))

test_evaluator = dict(type='AucMetric')
test_cfg = dict(type='TestLoop')

work_dir = './work_dirs/videomaev2/deepfake/rawframes2_FocalLoss/layer_weight_decay/only_train/random_margin/random_crop/'

load_from = (
    'https://download.openmmlab.com/mmaction/v1.0/recognition/videomaev2/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400/vit-base-p16_videomaev2-vit-g-dist-k710-pre_16x4x1_kinetics-400_20230510-3e7f93b2.pth'
)

randomness = dict(deterministic=True, diff_rank_seed=True, seed=1221)
