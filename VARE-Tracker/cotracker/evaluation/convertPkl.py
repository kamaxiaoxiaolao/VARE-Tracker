import os
import torch
import pickle
import numpy as np
from tqdm import tqdm


def pack_pth_to_tapvid_pkl(pth_dir, output_pkl_path):
    """将多个零散的 .pth 测试集文件打包成官方 TAP-Vid 格式的 .pkl"""
    print(f"📦 正在从 {pth_dir} 读取数据...")

    tapvid_dict = {}
    pth_files = [f for f in os.listdir(pth_dir) if f.endswith('.pth')]

    for f_name in tqdm(pth_files, desc="转换进度"):
        file_path = os.path.join(pth_dir, f_name)
        data = torch.load(file_path)

        vid_name = data["video_name"]

        # 1. 处理视频: (T, 3, H, W) Tensor -> (T, H, W, 3) numpy uint8
        # TAP-Vid 官方存储的是 uint8 格式以节省内存
        video_tensor = data["video"]
        if video_tensor.is_floating_point():
            # 假设原来是 0-255 的浮点数
            video_np = video_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        else:
            video_np = video_tensor.permute(0, 2, 3, 1).numpy()

        # 2. 处理轨迹: (T, N, 2) Tensor -> (N, T, 2) numpy float32
        # TAP-Vid 要求坐标归一化到 [0, 1]，除以 (W-1, H-1)
        H, W = video_np.shape[1], video_np.shape[2]
        points_np = data["gt_tracks"].permute(1, 0, 2).numpy().astype(np.float32)
        points_np /= np.array([W - 1, H - 1], dtype=np.float32)

        # 3. 处理遮挡: visibility (T, N) -> occluded (N, T) numpy bool
        # visibility 是 1 代表可见，0 代表遮挡；TAP-Vid 需要 True 代表遮挡
        vis_tensor = data["gt_visibility"]
        occluded_np = ~(vis_tensor.bool()).permute(1, 0).numpy()

        # 组装为官方需要的字典结构
        tapvid_dict[vid_name] = {
            "video": video_np,
            "points": points_np,
            "occluded": occluded_np
        }

    # 保存为 .pkl
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(tapvid_dict, f)

    print(f"\n✅ 打包完成！共处理 {len(pth_files)} 个视频。文件已保存至: {output_pkl_path}")

# 运行示例
pack_pth_to_tapvid_pkl(
    pth_dir="E:/Study/PY/workspace/co-tracker-main/CoTracker-Light/pths",
    output_pkl_path="F:/pkl/my.pkl"
)