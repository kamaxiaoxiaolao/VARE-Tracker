import os
import torch
import numpy as np
import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

from cotracker.datasets.utils import CoTrackerData


class DroneDataset(Dataset):
    """
    带标签的无人机数据集加载器。
    它会读取 YOLO 标签来确定无人机的位置，并只在无人机内部采样点。
    """

    def __init__(
            self,
            root_dir,
            seq_len=24,
            crop_size=(384, 512),
            traj_per_sample=384,  # 从训练命令中获取
            **kwargs
    ):
        super(DroneDataset, self).__init__()

        self.root_dir = root_dir
        self.seq_len = seq_len
        self.crop_size = tuple(crop_size)
        self.traj_per_sample = traj_per_sample

        # 定位到 train 文件夹
        train_path = os.path.join(self.root_dir, 'train')
        if not os.path.isdir(train_path):
            raise FileNotFoundError(f"错误: 在'{self.root_dir}'下找不到 'train' 文件夹。")

        # 获取所有视频文件夹
        self.video_folders = sorted([d for d in glob.glob(os.path.join(train_path, '*')) if os.path.isdir(d)])

        print(f"在 {train_path} 找到了 {len(self.video_folders)} 个视频片段。")

    def __len__(self):
        return len(self.video_folders)

    def getitem_helper(self, index):
        video_folder = self.video_folders[index]
        visible_folder = os.path.join(video_folder, 'visible')
        frame_paths = sorted(glob.glob(os.path.join(visible_folder, '*.jpg')))

        if len(frame_paths) < self.seq_len:
            return None, False

        start_frame_idx = np.random.randint(0, len(frame_paths) - self.seq_len + 1)
        selected_paths = frame_paths[start_frame_idx: start_frame_idx + self.seq_len]

        frames = []
        original_hw = None
        for frame_path in selected_paths:
            img = Image.open(frame_path).convert('RGB')
            if original_hw is None:
                original_hw = img.size[::-1]  # (H, W)
            frames.append(F.to_tensor(img))

        video = torch.stack(frames, dim=0)

        # --- !! 核心修改：读取标签并在框内采样 !! ---
        # 1. 找到第一帧对应的标签文件
        first_frame_path = selected_paths[0]
        label_file_name = os.path.basename(first_frame_path).replace('.jpg', '.txt')
        label_folder = os.path.join(video_folder, 'labels_yolo')  # 假设您的标签在 'labels_yolo' 文件夹
        label_path = os.path.join(label_folder, label_file_name)

        queries = None
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                # 只读取第一个边界框
                line = f.readline().strip()
                if line:
                    _, x_center, y_center, w, h = map(float, line.split())

                    # 将 YOLO 格式 (0-1, 相对坐标) 转换为像素坐标
                    H, W = original_hw
                    x_min = (x_center - w / 2) * W
                    y_min = (y_center - h / 2) * H
                    x_max = (x_center + w / 2) * W
                    y_max = (y_center + h / 2) * H

                    # 在边界框内部随机采样 N 个点
                    # np.random.uniform(low, high, size)
                    rand_x = np.random.uniform(x_min, x_max, self.traj_per_sample)
                    rand_y = np.random.uniform(y_min, y_max, self.traj_per_sample)

                    # 组合成 (N, 3) 的张量, t=0
                    queries_np = np.zeros((self.traj_per_sample, 3))
                    queries_np[:, 1] = rand_y
                    queries_np[:, 2] = rand_x

                    queries = torch.from_numpy(queries_np).float().unsqueeze(0)  # 添加 batch 维度

        if queries is None:
            # 如果第一帧没有标签，或者标签文件不存在，则跳过这个样本
            return None, False
        # --- 修改结束 ---

        # 视频预处理
        video_resized = F.resize(video, list(self.crop_size), antialias=True)

        # 同样需要对查询点坐标进行缩放
        scale_h = self.crop_size[0] / original_hw[0]
        scale_w = self.crop_size[1] / original_hw[1]
        queries[:, :, 1] *= scale_h
        queries[:, :, 2] *= scale_w

        sample = CoTrackerData(
            video=video_resized,
            query_points=queries,  # <--- 传递我们精心挑选的点
            trajectory=torch.ones(self.seq_len, self.traj_per_sample, 2),
            visibility=torch.ones(self.seq_len, self.traj_per_sample),
            valid=torch.ones(self.seq_len, self.traj_per_sample),
            seq_name=os.path.basename(video_folder),
        )

        return sample, True

    def __getitem__(self, index):
        sample, gotit = self.getitem_helper(index)
        if not gotit:
            # 创建一个假的空样本
            sample = CoTrackerData(
                video=torch.zeros((self.seq_len, 3, self.crop_size[0], self.crop_size[1])),
                query_points=torch.zeros((1, self.traj_per_sample, 3)),
            )
        return sample, gotit