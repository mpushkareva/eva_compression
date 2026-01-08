# video_dataloaders.py
import os
import random
from typing import List, Tuple, Callable, Optional, Literal, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video

# Reuse same normalization as images
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FrameSamplingMode = Literal["uniform", "random", "dense"]


def _get_video_train_frame_transform(
    img_size: int = 224,
    use_randaug: bool = True,
) -> Callable:
    """
    Returns a function that takes a list of PIL Images (frames) and returns a tensor:
      shape (C, T, H, W), normalized.
    The same spatial transforms are applied to all frames in a clip.
    """
    base_transforms = [
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaug:
        base_transforms.append(transforms.RandAugment())

    img_transform = transforms.Compose(
        base_transforms
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    def transform_clip(frames: Sequence["PIL.Image.Image"]) -> torch.Tensor:
        # Apply the same transform to each frame independently
        processed = [img_transform(frame) for frame in frames]  # (C, H, W)
        # Stack into (T, C, H, W) then permute to (C, T, H, W)
        clip = torch.stack(processed, dim=0)  # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
        return clip

    return transform_clip


def _get_video_val_frame_transform(
    img_size: int = 224,
    eval_resize: int = 256,
) -> Callable:
    """
    Validation transform for video frames.
    Resize shorter side to eval_resize, then center-crop img_size.
    """
    img_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    def transform_clip(frames: Sequence["PIL.Image.Image"]) -> torch.Tensor:
        processed = [img_transform(frame) for frame in frames]
        clip = torch.stack(processed, dim=0)  # (T, C, H, W)
        clip = clip.permute(1, 0, 2, 3)  # (C, T, H, W)
        return clip

    return transform_clip


def _sample_frame_indices(
    num_frames_in_video: int,
    num_samples: int,
    mode: FrameSamplingMode,
) -> List[int]:
    """Compute frame indices according to the sampling mode."""
    if num_frames_in_video <= 0:
        return []

    if mode == "dense":
        # Use all frames
        return list(range(num_frames_in_video))

    if num_frames_in_video < num_samples:
        # If video is too short, repeat frames
        if mode == "uniform":
            # Repeat uniformly
            step = num_frames_in_video / float(num_samples)
            indices = [min(int(step * i), num_frames_in_video - 1) for i in range(num_samples)]
        else:  # random
            indices = [random.randint(0, num_frames_in_video - 1) for _ in range(num_samples)]
        return indices

    if mode == "uniform":
        # Evenly spaced indices
        step = num_frames_in_video / float(num_samples)
        indices = [int(step * (i + 0.5)) for i in range(num_samples)]
        indices = [min(idx, num_frames_in_video - 1) for idx in indices]
        return indices

    if mode == "random":
        indices = sorted(random.sample(range(num_frames_in_video), num_samples))
        return indices

    raise ValueError(f"Unknown frame sampling mode: {mode}")


class VideoClassificationDataset(Dataset):
    """
    Generic video classification dataset, suitable for Kinetics / Something-Something style layouts.

    Two supported ways to specify data:
      1) Root + class folders:
           root / <label_name> / video_001.mp4
           root / <label_name> / video_002.mp4
         In this case, pass label_map=None and dataset will create a mapping from folder names.

      2) Explicit index file: each line: "<path_to_video> <class_id>"
         In this case, pass index_file path and label_map can be None
         (will use provided numeric class_id).
    """

    def __init__(
        self,
        root: str,
        index_file: Optional[str] = None,
        frame_transform: Optional[Callable] = None,
        num_frames: int = 10,
        sampling_mode: FrameSamplingMode = "uniform",
        video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".webm"),
    ) -> None:
        super().__init__()
        self.root = root
        self.index_file = index_file
        self.frame_transform = frame_transform
        self.num_frames = num_frames
        self.sampling_mode = sampling_mode
        self.video_exts = video_exts

        self.samples: List[Tuple[str, int]] = []  # (video_path, label_id)

        if index_file is not None:
            self._load_from_index_file()
        else:
            self._scan_root_directory()

    def _is_video_file(self, fname: str) -> bool:
        return any(fname.lower().endswith(ext) for ext in self.video_exts)

    def _scan_root_directory(self) -> None:
        """Scan root/class_name/video_file.* structure."""
        class_names = sorted(
            d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))
        )
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

        for cls_name in class_names:
            cls_dir = os.path.join(self.root, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if self._is_video_file(fname):
                    path = os.path.join(cls_dir, fname)
                    label = class_to_idx[cls_name]
                    self.samples.append((path, label))

    def _load_from_index_file(self) -> None:
        """Load (video_path, class_id) pairs from a simple text index file."""
        with open(self.index_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Allow spaces in path by assuming last token is label
                *path_tokens, label_str = line.split()
                rel_path = " ".join(path_tokens)
                video_path = os.path.join(self.root, rel_path)
                label = int(label_str)
                self.samples.append((video_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]

        # read_video returns: video(T, H, W, C), audio, info
        video, _, info = read_video(video_path, pts_unit="sec")
        num_frames = video.shape[0]

        if num_frames == 0:
            # Fallback: create a single black frame if video is unreadable
            # (T=1, H, W, C) will be created below
            raise RuntimeError(f"Video {video_path} has no frames.")

        indices = _sample_frame_indices(
            num_frames_in_video=num_frames,
            num_samples=self.num_frames,
            mode=self.sampling_mode,
        )

        # Select frames
        selected = video[indices] if len(indices) > 0 else video  # (T, H, W, C)

        # Convert to list of PIL Images for image-like transforms
        from torchvision.transforms.functional import to_pil_image

        frames = [to_pil_image(frame) for frame in selected]  # len T

        if self.frame_transform is not None:
            clip_tensor = self.frame_transform(frames)  # (C, T, H, W)
        else:
            # Minimal processing if no transform is provided
            to_tensor = transforms.ToTensor()
            clip_tensor_list = [to_tensor(f) for f in frames]
            clip_tensor = torch.stack(clip_tensor_list, dim=1)  # (C, T, H, W)

        return clip_tensor, label
        

def get_video_dataloaders(
    root: str,
    train_index_file: Optional[str] = None,
    val_index_file: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 8,
    num_frames: int = 10,
    sampling_mode: FrameSamplingMode = "uniform",
    img_size: int = 224,
    eval_resize: int = 256,
    use_randaug: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for generic video classification.

    Arguments:
        root: root folder that contains videos or paths referenced in index files.
        train_index_file: optional text file listing training videos and labels.
        val_index_file: optional text file listing validation videos and labels.
        If index files are None, dataset expects root/class_name/video.* structure.

    This function can be used for Kinetics or Something-Something with the proper index files.
    """
    train_frame_transform = _get_video_train_frame_transform(
        img_size=img_size,
        use_randaug=use_randaug,
    )
    val_frame_transform = _get_video_val_frame_transform(
        img_size=img_size,
        eval_resize=eval_resize,
    )

    train_ds = VideoClassificationDataset(
        root=root,
        index_file=train_index_file,
        frame_transform=train_frame_transform,
        num_frames=num_frames,
        sampling_mode=sampling_mode,
    )
    val_ds = VideoClassificationDataset(
        root=root,
        index_file=val_index_file,
        frame_transform=val_frame_transform,
        num_frames=num_frames,
        sampling_mode=sampling_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
