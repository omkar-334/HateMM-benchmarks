import gc

import torch
import torch.utils.data
import torchvision
from tqdm import tqdm

from preprocessing.video_dataset import VideoDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16


def pretrained_resnet152() -> torch.nn.Module:
    resnet152 = torchvision.models.resnet152(pretrained=True)
    resnet152.eval()
    for param in resnet152.parameters():
        param.requires_grad = False
    return resnet152


def process_single_video(video_id: str, resnet: torch.nn.Module, transform, progress_bar=None):
    """Process a single video and save its features."""
    try:
        dataset = VideoDataset(video_id=video_id, transform=transform)
        if not dataset.video_ids:
            return

        instance = dataset[0]
        frames = instance["frames"].to(DEVICE, non_blocking=True)

        video_features = torch.empty((len(frames), 2048), dtype=torch.float32, device=DEVICE)

        for start_idx in range(0, len(frames), BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, len(frames))
            batch = frames[start_idx:end_idx]

            with torch.no_grad():
                features = resnet(batch)
                video_features[start_idx:end_idx] = features

            if progress_bar:
                progress_bar.update(end_idx - start_idx)

        feature_path = dataset.features_file_path("resnet", "pool5", video_id)
        torch.save(video_features, feature_path)

        del video_features, frames
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        return


def save_resnet_features() -> None:
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    temp_dataset = VideoDataset(transform=transforms)
    video_ids = temp_dataset.video_ids
    total_frame_count = sum(temp_dataset.frame_count_by_video_id.values())
    del temp_dataset

    resnet = pretrained_resnet152().to(DEVICE)
    resnet.fc = torch.nn.Identity()

    with tqdm(total=total_frame_count, desc="Extracting ResNet features") as progress_bar:
        for video_id in video_ids:
            process_single_video(video_id, resnet, transforms, progress_bar)
            torch.cuda.empty_cache()
            gc.collect()
