# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import io
import os
import os.path as osp
import shutil
from typing import Dict, List, Optional, Union, Tuple, Callable, Sequence
from mmaction.structures import ActionDataSample
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.fileio import FileClient
import albumentations as alb
from mmaction.registry import TRANSFORMS
from torchvision.transforms import Normalize
import torch.nn.functional as F
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import InstanceData
import cv2
import random
import math

@TRANSFORMS.register_module()
class DeepfakeSampleFrames(BaseTransform):
    """Sample frames from the video.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Defaults to 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Defaults to False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Defaults to False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Defaults to 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Defaults to False.
        target_fps (optional, int): Convert input videos with arbitrary frame
            rates to the unified target FPS before sampling frames. If
            ``None``, the frame rate will not be adjusted. Defaults to
            ``None``.
    """

    def __init__(self,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 test_mode: bool = False,
                 keep_tail_frames: bool = False,
                 target_fps: Optional[int] = None,
                 **kwargs) -> None:

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.target_fps = target_fps
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames: int,
                         ori_clip_len: float) -> np.array:
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(num_frames - ori_clip_len + 1,
                                      size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int,
                        ori_clip_len: float) -> np.array:
        """Get clip offsets in test mode.

        If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        if self.clip_len == 1:  # 2D recognizer
            # assert self.frame_interval == 1
            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:  # 3D recognizer
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1
                # align test sample strategy with `PySlowFast` repo
                if self.target_fps is not None:
                    offset_between = np.floor(max_offset / float(num_segments))
                    clip_offsets = np.arange(num_clips) * offset_between
                else:
                    offset_between = max_offset / float(num_segments)
                    clip_offsets = np.arange(num_clips) * offset_between
                    clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)

        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio: float) -> float:
        """calculate length of clip segment for different strategy.

        Args:
            fps_scale_ratio (float): Scale ratio to adjust fps.
        """
        if self.target_fps is not None:
            # align test sample strategy with `PySlowFast` repo
            ori_clip_len = self.clip_len * self.frame_interval
            ori_clip_len = np.maximum(1, ori_clip_len * fps_scale_ratio)
        elif self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval

        return ori_clip_len

    def transform(self, results: dict) -> dict:
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        # if can't get fps, same value of `fps` and `target_fps`
        # will perform nothing
        fps = results.get('avg_fps')
        if self.target_fps is None or not fps:
            fps_scale_ratio = 1.0
        else:
            fps_scale_ratio = fps / self.target_fps
        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets = self._sample_clips(total_frames, ori_clip_len)

        if self.target_fps:
            frame_inds = clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len).astype(np.int32)
        else:
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(self.frame_interval,
                                                 size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + 0
        results['offset'] = 0
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        results['frames_path'] = [
            os.path.join(results['frame_dir'], p)
            for p in os.listdir(results['frame_dir']) if p.endswith('.png')
        ]
        results['frames_path'] = sorted(results['frames_path'])
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str
@TRANSFORMS.register_module()
class DeepfakeEvenSampleFrames(BaseTransform):
    """Sample frames from the video.

    Required Keys:

        - total_frames
        - start_index

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Defaults to 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Defaults to False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Defaults to False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Defaults to 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Defaults to False.
        target_fps (optional, int): Convert input videos with arbitrary frame
            rates to the unified target FPS before sampling frames. If
            ``None``, the frame rate will not be adjusted. Defaults to
            ``None``.
    """

    def __init__(self,
                 clip_len: int,
                 frame_interval: int = 1,
                 num_clips: int = 1,
                 temporal_jitter: bool = False,
                 twice_sample: bool = False,
                 out_of_bound_opt: str = 'loop',
                 test_mode: bool = False,
                 keep_tail_frames: bool = False,
                 target_fps: Optional[int] = None,
                 **kwargs) -> None:

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        self.target_fps = target_fps
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

    def _get_train_clips(self, num_frames: int,
                         ori_clip_len: float) -> np.array:
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int32)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(num_frames - ori_clip_len + 1,
                                      size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int32)

        return clip_offsets

    def _get_test_clips(self, num_frames: int,
                        ori_clip_len: float) -> np.array:
        """Get clip offsets in test mode.

        If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.
            ori_clip_len (float): length of original sample clip.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        if self.clip_len == 1:  # 2D recognizer
            # assert self.frame_interval == 1
            avg_interval = num_frames / float(self.num_clips)
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + avg_interval / 2.0
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:  # 3D recognizer
            max_offset = max(num_frames - ori_clip_len, 0)
            if self.twice_sample:
                num_clips = self.num_clips * 2
            else:
                num_clips = self.num_clips
            if num_clips > 1:
                num_segments = self.num_clips - 1
                # align test sample strategy with `PySlowFast` repo
                if self.target_fps is not None:
                    offset_between = np.floor(max_offset / float(num_segments))
                    clip_offsets = np.arange(num_clips) * offset_between
                else:
                    offset_between = max_offset / float(num_segments)
                    clip_offsets = np.arange(num_clips) * offset_between
                    clip_offsets = np.round(clip_offsets)
            else:
                clip_offsets = np.array([max_offset // 2])
        return clip_offsets

    def _sample_clips(self, num_frames: int, ori_clip_len: float) -> np.array:
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, ori_clip_len)
        else:
            clip_offsets = self._get_train_clips(num_frames, ori_clip_len)

        return clip_offsets

    def _get_ori_clip_len(self, fps_scale_ratio: float) -> float:
        """calculate length of clip segment for different strategy.

        Args:
            fps_scale_ratio (float): Scale ratio to adjust fps.
        """
        if self.target_fps is not None:
            # align test sample strategy with `PySlowFast` repo
            ori_clip_len = self.clip_len * self.frame_interval
            ori_clip_len = np.maximum(1, ori_clip_len * fps_scale_ratio)
        elif self.test_mode:
            ori_clip_len = (self.clip_len - 1) * self.frame_interval + 1
        else:
            ori_clip_len = self.clip_len * self.frame_interval

        return ori_clip_len

    def transform(self, results: dict) -> dict:
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        # if can't get fps, same value of `fps` and `target_fps`
        # will perform nothing
        fps = results.get('avg_fps')
        if self.target_fps is None or not fps:
            fps_scale_ratio = 1.0
        else:
            fps_scale_ratio = fps / self.target_fps
        ori_clip_len = self._get_ori_clip_len(fps_scale_ratio)
        clip_offsets = self._sample_clips(total_frames, ori_clip_len)

        if self.target_fps:
            frame_inds = clip_offsets[:, None] + np.linspace(
                0, ori_clip_len - 1, self.clip_len).astype(np.int32)
        else:
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(self.frame_interval,
                                                 size=len(frame_inds))
            frame_inds += perframe_offsets
            
        if frame_inds[0] % 2 != 0:
            if frame_inds[0] + 1 < total_frames:
                frame_inds += 1  
            else:
                frame_inds -= 1 
        frame_inds = frame_inds.reshape((-1, self.clip_len))
 
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + 0
        results['offset'] = 0
        results['frame_inds'] = frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        results['frames_path'] = [
            os.path.join(results['frame_dir'], p)
            for p in os.listdir(results['frame_dir']) if p.endswith('.png')
        ]
        results['frames_path'] = sorted(results['frames_path'])
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str

@TRANSFORMS.register_module()
class DeepfakeFrameOFDecode(BaseTransform):
    """Load and decode frames with given indices.

    Required Keys:

    - frame_dir
    - filename_tmpl
    - frame_inds
    - modality
    - offset (optional)

    Added Keys:

    - img
    - img_shape
    - original_shape

    Args:
        io_backend (str): IO backend where frames are stored.
            Defaults to ``'disk'``.
        decoding_backend (str): Backend used for image decoding.
            Defaults to ``'cv2'``.
    """

    def __init__(self,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results: dict) -> dict:
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        of_imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        cache = {}
        of_cache = {}
        for i, frame_idx in enumerate(results['frame_inds']):
            # Avoid loading duplicated frames
            if frame_idx in cache:
                imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                of_imgs.append(cp.deepcopy(of_imgs[of_cache[frame_idx]]))
                continue
            else:
                cache[frame_idx] = i
                of_cache[frame_idx] = i

            frame_idx += offset
            if modality == 'RGB':
                filepath = results['frames_path'][frame_idx]
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)

                of_filepath = results['frames_path'][frame_idx].replace(
                    'rawframes2', 'optical_flowformer_vis_aug')
                of_img_bytes = self.file_client.get(of_filepath)
                # Get frame with channel order RGB directly.
                of_cur_frame = mmcv.imfrombytes(of_img_bytes,
                                                channel_order='rgb')
                of_imgs.append(of_cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.append(np.stack([x_frame, y_frame], axis=-1))
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        results['of_imgs'] = of_imgs

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results


@TRANSFORMS.register_module()
class DeepfakeFrameDecode(BaseTransform):
    """Load and decode frames with given indices.

    Required Keys:

    - frame_dir
    - filename_tmpl
    - frame_inds
    - modality
    - offset (optional)

    Added Keys:

    - img
    - img_shape
    - original_shape

    Args:
        io_backend (str): IO backend where frames are stored.
            Defaults to ``'disk'``.
        decoding_backend (str): Backend used for image decoding.
            Defaults to ``'cv2'``.
    """

    def __init__(self,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results: dict) -> dict:
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        retinas = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)

        cache = {}
        retina_cache = {}
        for i, frame_idx in enumerate(results['frame_inds']):
            # Avoid loading duplicated frames
            if frame_idx in cache:
                imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                retinas.append(cp.deepcopy(retinas[retina_cache[frame_idx]]))
                continue
            else:
                cache[frame_idx] = i
                retina_cache[frame_idx] = i

            frame_idx += offset
            if modality == 'RGB':
                filepath = results['frames_path'][frame_idx]
                img_bytes = self.file_client.get(filepath)
                retina_path = filepath.replace('rawframes', 'retina').replace(
                    '.png', '.npy')
                retina = np.load(retina_path)[2:]
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
                retinas.append(retina)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.append(np.stack([x_frame, y_frame], axis=-1))
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        results['retinas'] = retinas

        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results


@TRANSFORMS.register_module()
class DeepfakeFrameDecode2(BaseTransform):
    """Load and decode frames with given indices.

    Required Keys:

    - frame_dir
    - filename_tmpl
    - frame_inds
    - modality
    - offset (optional)

    Added Keys:

    - img
    - img_shape
    - original_shape

    Args:
        io_backend (str): IO backend where frames are stored.
            Defaults to ``'disk'``.
        decoding_backend (str): Backend used for image decoding.
            Defaults to ``'cv2'``.
    """

    def __init__(self,
                 io_backend: str = 'disk',
                 decoding_backend: str = 'cv2',
                 **kwargs) -> None:
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def transform(self, results: dict) -> dict:
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        retinas = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
         
        cache = {}
        for i, frame_idx in enumerate(results['frame_inds']):
            # Avoid loading duplicated frames
            if frame_idx in cache:
                imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                continue
            else:
                cache[frame_idx] = i

            frame_idx += offset
            if modality == 'RGB':
                filepath = results['frames_path'][frame_idx]
                img_bytes = self.file_client.get(filepath)
                # Get frame with channel order RGB directly.
                cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                imgs.append(cur_frame)
            elif modality == 'Flow':
                x_filepath = osp.join(directory,
                                      filename_tmpl.format('x', frame_idx))
                y_filepath = osp.join(directory,
                                      filename_tmpl.format('y', frame_idx))
                x_img_bytes = self.file_client.get(x_filepath)
                x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                y_img_bytes = self.file_client.get(y_filepath)
                y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                imgs.append(np.stack([x_frame, y_frame], axis=-1))
            else:
                raise NotImplementedError

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        
        if 'token_label' in results:
            first_frame =  (results['frame_inds'][0])//2
            num_one_frame_token = 14*14
            results['token_label'] = results['token_label'][first_frame*num_one_frame_token:(first_frame+8)*num_one_frame_token]
            if 1 not in results['token_label']:
                results['label'] = 0
        # we resize the gt_bboxes and proposals to their real scale
        if 'gt_bboxes' in results:
            h, w = results['img_shape']
            scale_factor = np.array([w, h, w, h])
            gt_bboxes = results['gt_bboxes']
            gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
            results['gt_bboxes'] = gt_bboxes
            if 'proposals' in results and results['proposals'] is not None:
                proposals = results['proposals']
                proposals = (proposals * scale_factor).astype(np.float32)
                results['proposals'] = proposals

        return results


@TRANSFORMS.register_module()
class AlbumentationsColorJitter(BaseTransform):

    def transform(self, results):
        """Perform ColorJitter using Albumentations.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        num_clips, clip_len = results['num_clips'], results['clip_len']

        new_imgs = []
        transform = self.get_list_transforms()

        for i in range(num_clips):
            start, end = i * clip_len, (i + 1) * clip_len

            # Prepare the input dictionary for Albumentations
            img_dict = {}
            for idx, img in enumerate(imgs[start:end]):
                img = img.astype(np.uint8)
                if idx == 0:
                    img_dict[f"image"] = img
                else:
                    img_dict[f"image{str(idx).zfill(2)}"] = img

            # Apply the transformations
            transformed = transform(**img_dict)

            # Extract the transformed images
            for idx in range(clip_len):
                if idx == 0:
                    new_imgs.append(transformed[f"image"])
                else:
                    new_imgs.append(transformed[f"image{str(idx).zfill(2)}"])

        results['imgs'] = new_imgs
        return results

    def get_list_transforms(self):
        return alb.Compose(
            [
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3),
                    sat_shift_limit=(-0.3, 0.3),
                    val_shift_limit=(-0.3, 0.3),
                    p=0.3,
                ),
                alb.RandomBrightnessContrast(brightness_limit=(-0.7, 0.7),
                                             contrast_limit=(-0.7, 0.7),
                                             p=0.3),
                alb.ImageCompression(
                    quality_lower=40, quality_upper=100, p=0.5),
                alb.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                alb.ToGray(p=0.3),
            ],
            # defined additional target that key are image01 to image32 and the value are image
            additional_targets={
                "image01": "image",
                "image02": "image",
                "image03": "image",
                "image04": "image",
                "image05": "image",
                "image06": "image",
                "image07": "image",
                "image08": "image",
                "image09": "image",
                "image10": "image",
                "image11": "image",
                "image12": "image",
                "image13": "image",
                "image14": "image",
                "image15": "image",
                "image16": "image",
                "image17": "image",
                "image18": "image",
                "image19": "image",
                "image20": "image",
                "image21": "image",
                "image22": "image",
                "image23": "image",
                "image24": "image",
                "image25": "image",
                "image26": "image",
                "image27": "image",
                "image28": "image",
                "image29": "image",
                "image30": "image",
                "image31": "image",
            },
            p=1.0,
        )
@TRANSFORMS.register_module()
class AlbumentationsColorJitterRobust(BaseTransform):

    def transform(self, results):
        """Perform ColorJitter using Albumentations.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        imgs = results['imgs']
        num_clips, clip_len = results['num_clips'], results['clip_len']

        new_imgs = []
        transform = self.get_list_transforms()

        for i in range(num_clips):
            start, end = i * clip_len, (i + 1) * clip_len

            # Prepare the input dictionary for Albumentations
            img_dict = {}
            for idx, img in enumerate(imgs[start:end]):
                img = img.astype(np.uint8)
                if idx == 0:
                    img_dict[f"image"] = img
                else:
                    img_dict[f"image{str(idx).zfill(2)}"] = img

            # Apply the transformations
            transformed = transform(**img_dict)

            # Extract the transformed images
            for idx in range(clip_len):
                if idx == 0:
                    new_imgs.append(transformed[f"image"])
                else:
                    new_imgs.append(transformed[f"image{str(idx).zfill(2)}"])

        results['imgs'] = new_imgs
        return results

    def get_list_transforms(self):
        return alb.Compose(
            [
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3),
                    sat_shift_limit=(-0.3, 0.3),
                    val_shift_limit=(-0.3, 0.3),
                    p=0.3,
                ),
                alb.RandomBrightnessContrast(brightness_limit=(-0.7, 0.7),
                                             contrast_limit=(-0.7, 0.7),
                                             p=0.3),
                alb.ImageCompression(
                    quality_lower=40, quality_upper=100, p=0.5),
                alb.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                # alb.ToGray(p=0.3),
            ],
            # defined additional target that key are image01 to image32 and the value are image
            additional_targets={
                "image01": "image",
                "image02": "image",
                "image03": "image",
                "image04": "image",
                "image05": "image",
                "image06": "image",
                "image07": "image",
                "image08": "image",
                "image09": "image",
                "image10": "image",
                "image11": "image",
                "image12": "image",
                "image13": "image",
                "image14": "image",
                "image15": "image",
                "image16": "image",
                "image17": "image",
                "image18": "image",
                "image19": "image",
                "image20": "image",
                "image21": "image",
                "image22": "image",
                "image23": "image",
                "image24": "image",
                "image25": "image",
                "image26": "image",
                "image27": "image",
                "image28": "image",
                "image29": "image",
                "image30": "image",
                "image31": "image",
            },
            p=1.0,
        )

@TRANSFORMS.register_module()
class FaceTransform(BaseTransform):

    def get_similarity_transform_matrix(self, from_pts: torch.Tensor,
                                        to_pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            from_pts, to_pts: b x n x 2

        Returns:
            torch.Tensor: b x 3 x 3
        """
        mfrom = from_pts.mean(dim=1, keepdim=True)  # b x 1 x 2
        mto = to_pts.mean(dim=1, keepdim=True)  # b x 1 x 2

        a1 = (from_pts - mfrom).square().sum([1, 2], keepdim=False)  # b
        c1 = ((to_pts - mto) * (from_pts - mfrom)).sum([1, 2],
                                                       keepdim=False)  # b

        to_delta = to_pts - mto
        from_delta = from_pts - mfrom
        c2 = (to_delta[:, :, 0] * from_delta[:, :, 1] -
              to_delta[:, :, 1] * from_delta[:, :, 0]).sum([1],
                                                           keepdim=False)  # b

        a = c1 / a1
        b = c2 / a1
        dx = mto[:, 0, 0] - a * mfrom[:, 0, 0] - b * mfrom[:, 0, 1]  # b
        dy = mto[:, 0, 1] + b * mfrom[:, 0, 0] - a * mfrom[:, 0, 1]  # b

        ones_pl = torch.ones_like(a1)
        zeros_pl = torch.zeros_like(a1)

        return torch.stack([
            a,
            b,
            dx,
            -b,
            a,
            dy,
            zeros_pl,
            zeros_pl,
            ones_pl,
        ],
                           dim=-1).reshape(-1, 3, 3)

    def _standard_face_pts(self):
        pts = torch.tensor([
            196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0,
            360.4
        ],
                           dtype=torch.float32) / 256.0 - 1.0
        return torch.reshape(pts, (5, 2))

    def get_face_align_matrix(self,
                              face_pts: torch.Tensor,
                              target_shape: Tuple[int, int],
                              target_face_scale: float = 1.0,
                              offset_xy: Optional[Tuple[float, float]] = None,
                              target_pts: Optional[torch.Tensor] = None):

        if target_pts is None:
            with torch.no_grad():
                std_pts = self._standard_face_pts().to(face_pts)  # [-1 1]
                h, w, *_ = target_shape
                target_pts = (std_pts * target_face_scale + 1) * \
                    torch.tensor([w-1, h-1]).to(face_pts) / 2.0
                if offset_xy is not None:
                    target_pts[:, 0] += offset_xy[0]
                    target_pts[:, 1] += offset_xy[1]
        else:
            target_pts = target_pts.to(face_pts)

        if target_pts.dim() == 2:
            target_pts = target_pts.unsqueeze(0)
        if target_pts.size(0) == 1:
            target_pts = target_pts.expand_as(face_pts)

        assert target_pts.shape == face_pts.shape

        return self.get_similarity_transform_matrix(face_pts, target_pts)

    def get_std_points_xray(self, out_size=256, mid_size=500):
        std_points_256 = np.array([
            [85.82991, 85.7792],
            [169.0532, 84.3381],
            [127.574, 137.0006],
            [90.6964, 174.7014],
            [167.3069, 173.3733],
        ])
        std_points_256[:, 1] += 30
        old_size = 256
        mid = mid_size / 2
        new_std_points = std_points_256 - old_size / 2 + mid
        target_pts = new_std_points * out_size / mid_size
        target_pts = torch.from_numpy(target_pts).float()
        return target_pts

    def _safe_arctanh(self,
                      x: torch.Tensor,
                      eps: float = 0.001) -> torch.Tensor:
        return torch.clamp(x, -1 + eps, 1 - eps).arctanh()

    def inverted_tanh_warp_transform(self, coords: torch.Tensor,
                                     matrix: torch.Tensor, warp_factor: float,
                                     warped_shape: Tuple[int, int]):
        """ Inverted tanh-warp function.

        Args:
            coords (torch.Tensor): b x n x 2 (x, y). The transformed coordinates.
            matrix: b x 3 x 3. A matrix that transforms un-normalized coordinates 
                from the original image to the aligned yet not-warped image.
            warp_factor (float): The warp factor. 
                0 means linear transform, 1 means full tanh warp.
            warped_shape (tuple): [height, width].

        Returns:
            torch.Tensor: b x n x 2 (x, y). The original coordinates.
        """
        h, w, *_ = warped_shape
        w_h = torch.tensor([[w, h]]).to(coords)

        if warp_factor > 0:
            # normalize coordinates to [-1, +1]
            coords = coords / w_h * 2 - 1

            nl_part1 = coords > 1.0 - warp_factor
            nl_part2 = coords < -1.0 + warp_factor

            ret_nl_part1 = self._safe_arctanh(
                (coords - 1.0 + warp_factor) /
                warp_factor) * warp_factor + \
                1.0 - warp_factor
            ret_nl_part2 = self._safe_arctanh(
                (coords + 1.0 - warp_factor) /
                warp_factor) * warp_factor - \
                1.0 + warp_factor

            coords = torch.where(nl_part1, ret_nl_part1,
                                 torch.where(nl_part2, ret_nl_part2, coords))

            # denormalize
            coords = (coords + 1) / 2 * w_h

        coords_homo = torch.cat(
            [coords, torch.ones_like(coords[:, :, [0]])], dim=-1)  # b x n x 3

        inv_matrix = torch.linalg.inv(matrix)  # b x 3 x 3
        coords_homo = torch.bmm(coords_homo,
                                inv_matrix.permute(0, 2, 1))  # b x n x 3
        return coords_homo[:, :, :2] / coords_homo[:, :, [2, 2]]

    def make_tanh_warp_grid(self, matrix: torch.Tensor, warp_factor: float,
                            warped_shape: Tuple[int, int],
                            orig_shape: Tuple[int, int]):
        """
        Args:
            matrix: bx3x3 matrix.
            warp_factor: The warping factor. `warp_factor=1.0` represents a vanilla Tanh-warping, 
            `warp_factor=0.0` represents a cropping.
            warped_shape: The target image shape to transform to.

        Returns:
            torch.Tensor: b x h x w x 2 (x, y).
        """
        orig_h, orig_w, *_ = orig_shape
        w_h = torch.tensor([orig_w, orig_h]).to(matrix).reshape(1, 1, 1, 2)
        return self._forge_grid(
            matrix.size(0), matrix.device, warped_shape,
            lambda coords: self.inverted_tanh_warp_transform(
                coords=coords,
                matrix=matrix,
                warp_factor=warp_factor,
                warped_shape=warped_shape)) / w_h * 2 - 1

    def _meshgrid(self, h, w) -> Tuple[torch.Tensor, torch.Tensor]:
        yy, xx = torch.meshgrid(
            torch.arange(h).float(),
            torch.arange(w).float())
        return yy + 0.5, xx + 0.5

    def _forge_grid(
        self, batch_size: int, device: torch.device,
        output_shape: Tuple[int, int], fn: Callable[[torch.Tensor],
                                                    torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forge transform maps with a given function `fn`.

        Args:
            output_shape (tuple): (b, h, w, ...).
            fn (Callable[[torch.Tensor], torch.Tensor]): The function that accepts 
                a bxnx2 array and outputs the transformed bxnx2 array. Both input 
                and output store (x, y) coordinates.

        Note: 
            both input and output arrays of `fn` should store (y, x) coordinates.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Two maps `X` and `Y`, where for each 
                pixel (y, x) or coordinate (x, y),
                `(X[y, x], Y[y, x]) = fn([x, y])`
        """
        h, w, *_ = output_shape
        yy, xx = self._meshgrid(h, w)  # h x w
        yy = yy.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        xx = xx.unsqueeze(0).expand(batch_size, -1, -1).to(device)

        in_xxyy = torch.stack([xx, yy], dim=-1).reshape([batch_size, h * w,
                                                         2])  # (h x w) x 2
        out_xxyy: torch.Tensor = fn(in_xxyy)  # (h x w) x 2
        return out_xxyy.reshape(batch_size, h, w, 2)

    def transform(self, results):
        images = results["imgs"]
        retinas = results["retinas"]
        new_images = []

        for image, retina in zip(images, retinas):
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            retina = torch.tensor(retina, dtype=torch.float32).unsqueeze(0)
            image = image.float() / 255.0
            _, _, h, w = image.shape

            matrix = self.get_face_align_matrix(
                face_pts=retina,
                target_shape=(224, 224),
                target_pts=self.get_std_points_xray(out_size=224,
                                                    mid_size=500))
            grid = self.make_tanh_warp_grid(matrix=matrix,
                                            orig_shape=(h, w),
                                            warped_shape=(224, 224),
                                            warp_factor=0.0)

            w_images = F.grid_sample(image,
                                     grid,
                                     mode="bilinear",
                                     align_corners=False)
            # w_images = self.normalize(w_images)
            new_images.append(
                w_images.squeeze(0).permute(1, 2, 0).numpy() * 255)

        results["imgs"] = new_images

        return results


@TRANSFORMS.register_module()
class FormatShapeDeepfakeOF(BaseTransform):
    """Format final imgs shape to the given input_format.

    Required keys:

        - imgs (optional)
        - heatmap_imgs (optional)
        - modality (optional)
        - num_clips
        - clip_len

    Modified Keys:

        - imgs

    Added Keys:

        - input_shape
        - heatmap_input_shape (optional)

    Args:
        input_format (str): Define the final data format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, input_format: str, collapse: bool = False) -> None:
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in [
                'NCTHW', 'NCHW', 'NCTHW_Heatmap', 'NPTCHW'
        ]:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results: Dict) -> Dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        if not isinstance(results['of_imgs'], np.ndarray):
            results['of_imgs'] = np.array(results['of_imgs'])

        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * T
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            if 'imgs' in results:
                imgs = results['imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

            if 'of_imgs' in results:
                imgs = results['of_imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x C x H x W
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['of_imgs'] = imgs
                results['of_input_shape'] = imgs.shape

        elif self.input_format == 'NCTHW_Heatmap':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x C x H x W
            imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
            # N_crops x N_clips x C x T x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x T x H x W
            # M' = N_crops x N_clips
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW':
            imgs = results['imgs']
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            if 'modality' in results and results['modality'] == 'Flow':
                clip_len = results['clip_len']
                imgs = imgs.reshape((-1, clip_len * imgs.shape[1]) +
                                    imgs.shape[2:])
            # M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x T
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        if self.collapse:
            assert results['imgs'].shape[0] == 1
            results['imgs'] = results['imgs'].squeeze(0)
            results['input_shape'] = results['imgs'].shape

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str
    
@TRANSFORMS.register_module()
class PackTokenInputs(BaseTransform):
    """Pack the inputs data.

    Args:
        collect_keys (tuple[str], optional): The keys to be collected
            to ``packed_results['inputs']``. Defaults to ``
        meta_keys (Sequence[str]): The meta keys to saved in the
            `metainfo` of the `data_sample`.
            Defaults to ``('img_shape', 'img_key', 'video_id', 'timestamp')``.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
    """

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_labels': 'labels',
    }

    def __init__(
            self,
            collect_keys: Optional[Tuple[str]] = None,
            meta_keys: Sequence[str] = ('img_shape', 'img_key', 'video_id',
                                        'timestamp'),
            algorithm_keys: Sequence[str] = (),
    ) -> None:
        self.collect_keys = collect_keys
        self.meta_keys = meta_keys
        self.algorithm_keys = algorithm_keys

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PackActionInputs`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        packed_results = dict()
        if self.collect_keys is not None:
            packed_results['inputs'] = dict()
            for key in self.collect_keys:
                packed_results['inputs'][key] = to_tensor(results[key])
        else:
            if 'imgs' in results and 'of_imgs' in results:
                imgs = results['imgs']
                of_imgs = results['of_imgs']
                packed_results['inputs'] = {
                    'imgs': to_tensor(imgs).float(),
                    'of_imgs': to_tensor(of_imgs).float()
                }
            elif 'imgs' in results:
                imgs = results['imgs']
                packed_results['inputs'] = to_tensor(imgs).float() / 255.0
            elif 'heatmap_imgs' in results:
                heatmap_imgs = results['heatmap_imgs']
                packed_results['inputs'] = to_tensor(heatmap_imgs)
            elif 'keypoint' in results:
                keypoint = results['keypoint']
                packed_results['inputs'] = to_tensor(keypoint)
            elif 'audios' in results:
                audios = results['audios']
                packed_results['inputs'] = to_tensor(audios)
            elif 'text' in results:
                text = results['text']
                packed_results['inputs'] = to_tensor(text)
            else:
                raise ValueError(
                    'Cannot get `imgs`, `keypoint`, `heatmap_imgs`, '
                    '`audios` or `text` in the input dict of '
                    '`PackActionInputs`.')

        data_sample = ActionDataSample()

        if 'gt_bboxes' in results:
            instance_data = InstanceData()
            for key in self.mapping_table.keys():
                instance_data[self.mapping_table[key]] = to_tensor(
                    results[key])
            data_sample.gt_instances = instance_data

            if 'proposals' in results:
                data_sample.proposals = InstanceData(
                    bboxes=to_tensor(results['proposals']))

        if 'label' in results:
            data_sample.set_gt_label(results['label'])
            
        if 'token_label' in results:
            instance_data = InstanceData()
            instance_data['token_label'] = to_tensor(results['token_label'])
            data_sample.gt_instances = instance_data

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(collect_keys={self.collect_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackDeepfakeInputs(BaseTransform):
    """Pack the inputs data.

    Args:
        collect_keys (tuple[str], optional): The keys to be collected
            to ``packed_results['inputs']``. Defaults to ``
        meta_keys (Sequence[str]): The meta keys to saved in the
            `metainfo` of the `data_sample`.
            Defaults to ``('img_shape', 'img_key', 'video_id', 'timestamp')``.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
    """

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_labels': 'labels',
    }

    def __init__(
            self,
            collect_keys: Optional[Tuple[str]] = None,
            meta_keys: Sequence[str] = ('img_shape', 'img_key', 'video_id',
                                        'timestamp'),
            algorithm_keys: Sequence[str] = (),
    ) -> None:
        self.collect_keys = collect_keys
        self.meta_keys = meta_keys
        self.algorithm_keys = algorithm_keys

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PackActionInputs`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        packed_results = dict()
        if self.collect_keys is not None:
            packed_results['inputs'] = dict()
            for key in self.collect_keys:
                packed_results['inputs'][key] = to_tensor(results[key])
        else:
            if 'imgs' in results and 'of_imgs' in results:
                imgs = results['imgs']
                of_imgs = results['of_imgs']
                packed_results['inputs'] = {
                    'imgs': to_tensor(imgs).float(),
                    'of_imgs': to_tensor(of_imgs).float()
                }
            elif 'imgs' in results:
                imgs = results['imgs']
                packed_results['inputs'] = to_tensor(imgs).float() / 255.0
            elif 'heatmap_imgs' in results:
                heatmap_imgs = results['heatmap_imgs']
                packed_results['inputs'] = to_tensor(heatmap_imgs)
            elif 'keypoint' in results:
                keypoint = results['keypoint']
                packed_results['inputs'] = to_tensor(keypoint)
            elif 'audios' in results:
                audios = results['audios']
                packed_results['inputs'] = to_tensor(audios)
            elif 'text' in results:
                text = results['text']
                packed_results['inputs'] = to_tensor(text)
            else:
                raise ValueError(
                    'Cannot get `imgs`, `keypoint`, `heatmap_imgs`, '
                    '`audios` or `text` in the input dict of '
                    '`PackActionInputs`.')

        data_sample = ActionDataSample()

        if 'gt_bboxes' in results:
            instance_data = InstanceData()
            for key in self.mapping_table.keys():
                instance_data[self.mapping_table[key]] = to_tensor(
                    results[key])
            data_sample.gt_instances = instance_data

            if 'proposals' in results:
                data_sample.proposals = InstanceData(
                    bboxes=to_tensor(results['proposals']))

        if 'label' in results:
            data_sample.set_gt_label(results['label'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(collect_keys={self.collect_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class Distortion(BaseTransform):

    def __init__(
            self,
            distor_type: str,
            level: int
    ) -> None:
        self.distor_type = distor_type
        self.level = level
        self.dist_param = self.get_distortion_parameter(distor_type, level)
        self.dist_function = self.get_distortion_function(distor_type)
        

    #============================distortion function 
    def rgb2ycbcr(self, img_rgb):
        img_rgb = img_rgb.astype(np.float32)
        img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCR_CB)
        img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
        # to [16/255, 235/255]
        img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
        # to [16/255, 240/255]
        img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0

        return img_ycbcr


    def ycbcr2bgr(self, img_ycbcr):
        img_ycbcr = img_ycbcr.astype(np.float32)
        # to [0, 1]
        img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
        # to [0, 1]
        img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
        img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
        img_rgb = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2RGB)

        return img_rgb


    def color_saturation(self, img, param):
        ycbcr = self.rgb2ycbcr(img)
        ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
        ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
        img = self.ycbcr2bgr(ycbcr).astype(np.uint8)

        return img


    def color_contrast(self, img, param):
        img = img.astype(np.float32) * param
        img = img.astype(np.uint8)

        return img


    def block_wise(self, img, param):
        width = 8
        block = np.ones((width, width, 3)).astype(int) * 128
        param = min(img.shape[0], img.shape[1]) // 224 * param
        for i in range(param):
            r_w = random.randint(0, img.shape[1] - 1 - width)
            r_h = random.randint(0, img.shape[0] - 1 - width)
            img[r_h:r_h + width, r_w:r_w + width, :] = block

        return img


    def gaussian_noise_color(self, img, param):
        ycbcr = self.rgb2ycbcr(img) / 255
        size_a = ycbcr.shape
        b = (ycbcr + math.sqrt(param) *
            np.random.randn(size_a[0], size_a[1], size_a[2])) * 255
        b = self.ycbcr2bgr(b)
        img = np.clip(b, 0, 255).astype(np.uint8)

        return img


    def gaussian_blur(self, img, param):
        img = cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)

        return img


    def jpeg_compression(self, img, param):
        h, w, _ = img.shape
        s_h = h // param
        s_w = w // param
        img = cv2.resize(img, (s_w, s_h))
        img = cv2.resize(img, (w, h))

        return img



    #============================distortion function 

    def get_distortion_parameter(self, type, level):
        param_dict = dict()  # a dict of list
        param_dict['CS'] = [0.4, 0.3, 0.2, 0.1, 0.0]  # smaller, worse
        param_dict['CC'] = [0.85, 0.725, 0.6, 0.475, 0.35]  # smaller, worse
        param_dict['BW'] = [16, 32, 48, 64, 80]  # larger, worse
        param_dict['GNC'] = [0.001, 0.002, 0.005, 0.01, 0.05]  # larger, worse
        param_dict['GB'] = [7, 9, 13, 17, 21]  # larger, worse
        param_dict['JPEG'] = [2, 3, 4, 5, 6]  # larger, worse
        param_dict['VC'] = [30, 32, 35, 38, 40]  # larger, worse

        # level starts from 1, list starts from 0
        return param_dict[type][level - 1]


    def get_distortion_function(self, type):
        func_dict = dict()  # a dict of function
        func_dict['CS'] = self.color_saturation
        func_dict['CC'] = self.color_contrast
        func_dict['BW'] = self.block_wise
        func_dict['GNC'] = self.gaussian_noise_color
        func_dict['GB'] = self.gaussian_blur
        func_dict['JPEG'] = self.jpeg_compression
        # func_dict['VC'] = self.frame_compression

        return func_dict[type]

    def transform(self, results):
        imgs = results['imgs']
        num_clips, clip_len = results['num_clips'], results['clip_len']

        new_imgs = []

        for i in range(num_clips):
            start, end = i * clip_len, (i + 1) * clip_len
            for idx, img in enumerate(imgs[start:end]):
                new_imgs.append(self.dist_function(img, self.dist_param))

        results['imgs'] = new_imgs
        return results
