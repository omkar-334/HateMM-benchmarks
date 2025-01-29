import json
import os
from typing import Callable, Dict

import PIL.Image
import torch
import torch.utils.data


class VideoDataset(torch.utils.data.Dataset):
    FRAMES_DIR_PATH = "frames"  # Input folder path
    FEATURES_DIR_PATH = "features"  # Output fodler path
    FRAME_SAMPLE_RATE = 20  # Take every 20th frame

    def __init__(
        self,
        video_id: str = None,
        transform: Callable = None,
        videos_data_path: str = "transcriptions.json",
        check_missing_videos: bool = False,
    ) -> None:
        self.transform = transform
        os.makedirs(self.FEATURES_DIR_PATH, exist_ok=True)

        with open(videos_data_path) as file:
            videos_data_dict = json.load(file)

        if video_id is not None:
            if video_id in videos_data_dict:
                self.video_ids = [video_id]
            else:
                raise ValueError(f"Video ID {video_id} not found in dataset")
        else:
            self.video_ids = list(videos_data_dict)

        self.frame_count_by_video_id = {}
        for vid in self.video_ids:
            video_folder_path = self._video_folder_path(vid)
            if not os.path.exists(video_folder_path):
                if check_missing_videos:
                    raise FileNotFoundError(f"Directory {video_folder_path} not found")
                continue
            total_frames = len(os.listdir(video_folder_path))
            sampled_frames = (total_frames + self.FRAME_SAMPLE_RATE - 1) // self.FRAME_SAMPLE_RATE
            self.frame_count_by_video_id[vid] = sampled_frames

    @staticmethod
    def _video_folder_path(video_id: str) -> str:
        return os.path.join(VideoDataset.FRAMES_DIR_PATH, video_id)

    @staticmethod
    def features_file_path(model_name: str, layer_name: str, video_id: str) -> str:
        return os.path.join(VideoDataset.FEATURES_DIR_PATH, f"{video_id}_{model_name}_{layer_name}.pt")

    def __getitem__(self, index) -> Dict[str, object]:
        video_id = self.video_ids[index]
        frames = []
        video_folder_path = self._video_folder_path(video_id)

        all_frames = sorted(os.listdir(video_folder_path))
        sampled_frames = all_frames[:: self.FRAME_SAMPLE_RATE]

        for frame_file_name in sampled_frames:
            frame = PIL.Image.open(os.path.join(video_folder_path, frame_file_name))
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        frames = torch.stack(frames)
        return {"id": video_id, "frames": frames}

    def __len__(self) -> int:
        return len(self.video_ids)
