import os
import random
import math
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

from basicsr.data import degradations as degradations
from basicsr.data.transforms import augment
from basicsr.utils import img2tensor

LIGHT_COND = ["%02d" % i for i in range(20)]

ANGLES_EXTREME = ["11_0", "12_0", "09_0", "19_1", "08_1", "20_0", "01_0", "24_0"]
ANGLES_MODERATE = ["08_0", "13_0", "14_0", "05_0", "04_1", "19_0"]

GT_ANGLES_MODERATE = ["08_0", "19_0"]
GT_ANGLES_FRONTAL = ["05_1", "05_1"]


class MultiPIEDataset(Dataset):
    @staticmethod
    def color_jitter(images, shift):
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)

        for idx in range(len(images)):
            images[idx] = images[idx] + jitter_val
            images[idx] = np.clip(images[idx], 0, 1)

        return images

    def __init__(
        self,
        dataroot: str,
        phase="train",
        res=128,
        use_blind=True,  # using blind degradation
    ):
        super().__init__()
        self.dataroot = os.path.join(dataroot, phase)
        self.res = res
        self.use_blind = use_blind

        self.input_paths = []
        self.input_angles = []
        self.gt_paths = []
        self.gt_patch_paths = []

        angles = [*ANGLES_EXTREME, *ANGLES_MODERATE]
        gt_angles = GT_ANGLES_FRONTAL

        for pid in sorted(os.listdir(self.dataroot)):
            for idx, angle in enumerate(angles):
                for light in LIGHT_COND:
                    gt_angle = gt_angles[0] if idx < len(angles) // 2 else gt_angles[1]
                    gt_path = os.path.join(
                        self.dataroot, pid, gt_angle, "%s.png" % light
                    )
                    input_path = os.path.join(
                        self.dataroot, pid, angle, "%s.png" % light
                    )
                    if all(map(os.path.exists, [gt_path, input_path])):
                        self.input_paths.append(input_path)
                        self.gt_paths.append(gt_path)
                        """
                        if self.use_patch:
                            gt_patch_path = os.path.join(
                                self.dataroot, pid, gt_angle, "%s_patch.png" % light
                            )
                            self.gt_patch_paths.append(gt_patch_path)
                        """

    def __getitem__(self, index):
        input_image = cv2.imread(self.input_paths[index])
        gt_image = cv2.imread(self.gt_paths[index])

        input_image = cv2.resize(
            input_image, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC
        )
        gt_image = cv2.resize(
            gt_image, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC
        )

        # random horizontal flip
        input_image, _status = augment(
            input_image, hflip=True, rotation=False, return_status=True
        )
        gt_image, _status = augment(
            gt_image, hflip=True, rotation=False, return_status=True
        )

        lq_nf = input_image.astype(np.float32) / 255.0
        lq_ft = gt_image.astype(np.float32) / 255.0
        
        # backup the HQ pair
        hq_nf = lq_nf.copy()
        hq_ft = lq_ft.copy()

        if self.use_blind:
            # blur
            cur_kernel_size = random.randint(19, 20) * 2 + 1
            kernel = degradations.random_mixed_kernels(
                ["iso", "aniso"],
                [0.5, 0.5],
                cur_kernel_size,    # 41 for DiffBIR
                [0.1, 2.0], # [0.1, 12] for DiffBIR
                [0.1, 2.0], # [0.1, 12] for DiffBIR
                [-math.pi, math.pi],
                noise_range=None,
            )
            lq_nf = cv2.filter2D(lq_nf, -1, kernel)
            lq_ft = cv2.filter2D(lq_ft, -1, kernel)

            # downsample
            scale = np.random.uniform(0.8, 8.0) # [1, 12] for DiffBIR
            lq_nf = cv2.resize(
                lq_nf,
                (int(self.res // scale), int(self.res // scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            lq_ft = cv2.resize(
                lq_ft,
                (int(self.res // scale), int(self.res // scale)),
                interpolation=cv2.INTER_LINEAR,
            )

            # noise
            sigma = np.random.uniform(0, 10) # [0, 15] for DiffBIR
            noise = np.float32(np.random.randn(*(lq_nf.shape))) * sigma / 255.
            lq_nf = np.clip(lq_nf + noise, 0, 1)
            lq_ft = np.clip(lq_ft + noise, 0, 1)
            
            # jpeg compression
            quality = np.random.uniform(80, 100) # [30, 100] for DiffBIR
            lq_nf = degradations.add_jpg_compression(lq_nf, quality)
            lq_ft = degradations.add_jpg_compression(lq_ft, quality)

            # resize to original size
            lq_nf = cv2.resize(
                lq_nf, (self.res, self.res), interpolation=cv2.INTER_LINEAR
            )
            lq_ft = cv2.resize(
                lq_ft, (self.res, self.res), interpolation=cv2.INTER_LINEAR
            )

            # random color jitter
            if np.random.uniform() < 0.5:
                lq_nf, lq_ft, hq_nf, hq_ft = self.color_jitter([lq_nf, lq_ft, hq_nf, hq_ft], 0.05)

            # random to gray (only for lq)
            if np.random.uniform() < 0.008:
                lq_nf = cv2.cvtColor(lq_nf, cv2.COLOR_BGR2GRAY)
                lq_nf = np.tile(lq_nf[:, :, None], [1, 1, 3])
                lq_ft = cv2.cvtColor(lq_ft, cv2.COLOR_BGR2GRAY)
                lq_ft = np.tile(lq_ft[:, :, None], [1, 1, 3])

        else:
            lq_nf = cv2.resize(
                lq_nf,
                dsize=(self.res // 4, self.res // 4),
                interpolation=cv2.INTER_CUBIC,
            )
            lq_nf = cv2.resize(
                lq_nf, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC
            )
            
            lq_ft = cv2.resize(
                lq_ft,
                dsize=(self.res // 4, self.res // 4),
                interpolation=cv2.INTER_CUBIC,
            )
            lq_ft = cv2.resize(
                lq_ft, dsize=(self.res, self.res), interpolation=cv2.INTER_CUBIC
            )

        # BGR to RGB, HWC to CHW, numpy to tensor
        lq_ft, lq_nf = img2tensor(
            [lq_ft, lq_nf], bgr2rgb=True, float32=True
        )
        hq_ft, hq_nf = img2tensor(
            [hq_ft, hq_nf], bgr2rgb=True, float32=True
        )

        # round and clip
        lq_nf = torch.clamp((lq_nf * 255.0).round(), 0, 255) / 255.0
        lq_ft = torch.clamp((lq_ft * 255.0).round(), 0, 255) / 255.0

        """
        if self.use_patch:  # unused
            gt_patch = to_tensor(
                Image.open(self.gt_patch_paths[index])
                .convert("RGB")
                .resize((self.res, self.res), Image.Resampling.BICUBIC)
            )

            return input_image, gt_image, gt_patch
        """

        return lq_nf, lq_ft, hq_nf, hq_ft

    def __len__(self):
        return len(self.gt_paths)

