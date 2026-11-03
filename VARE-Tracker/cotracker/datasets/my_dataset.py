import glob  # 确保在文件顶部导入 glob
import torch
import numpy as np
import os
class RealDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            crop_size=(384, 512),
            seq_len=24,
            traj_per_sample=768,
            random_frame_rate=False,
            random_seq_len=False,
            random_resize=False,
            limit_samples=10000,
            dataset_root=None,  # 新增的参数，非常重要！
            data_splits=[0],  # 保留以兼容旧接口，但不再使用
    ):
        super(RealDataset, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)

        # 1. 检查 dataset_root 是否提供
        if dataset_root is None:
            raise ValueError("dataset_root must be provided for RealDataset.")

        print(f"Searching for videos in: {dataset_root}")

        # 2. 自动查找所有视频文件
        # 使用 glob 查找常见的视频格式
        video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
        filelist_all = []
        for ext in video_extensions:
            filelist_all.extend(glob.glob(os.path.join(dataset_root, ext)))

        print(f"Found {len(filelist_all)} videos.")
        if len(filelist_all) == 0:
            raise FileNotFoundError(f"No videos found in {dataset_root}. Please check the path.")

        # 3. 应用 limit_samples
        self.filelist = filelist_all[:limit_samples]
        print(f"Using {len(self.filelist)} videos for training.")

        # 4. 保留其他必要的属性
        self.traj_per_sample = traj_per_sample
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.random_frame_rate = random_frame_rate
        self.random_resize = random_resize
        self.random_seq_len = random_seq_len