import cv2
import torch
import os
import pandas as pd

class VideoWriter():
    def __init__(
            self,
            output_path,
            fourcc,
            fps,
            shape
    ):
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, shape)
        self.type = None

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
aa

class PiperDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str | None = None,
            episode_num: int = None,
            episode_len: int = None,
            create_video : bool = True,
            fps: int = 30,
    ):
        self.root = root
        self.episode_num = episode_num
        self.create_video = create_video
        self.fps = fps
        self.episode_len = episode_len

        if not os.path.isdir(self.root):
            self.create_root_dir()

        self.episode_file = "episode.pickle"
        self.episode_path = os.path.join(self.root, str(self.episode_num))
        self.episode = []

        self.num_frames = self.fps * self.episode_len

    def save_episode(self):
        if not os.path.isdir(self.episode_path):
            self.create_episode_dir()

        for item in self.episode:
            for key in item:
                if isinstance(item[key], torch.Tensor):
                    item[key] = item[key].tolist()

        df = pd.DataFrame(self.episode)
        df.to_pickle(self.episode_file)

        if self.create_video:
            self.save_video()

    def add_frame(self, frame):
        self.episode.append(frame)

    def create_root_dir(self):
        os.makedirs(self.root, exist_ok=True)

    def create_episode_dir(self):
        os.makedirs(self.episode_path, exist_ok=True)

    def save_video(self):
        height, width = self.episode[0]["observation.images.wrist"].shape[:2]
        video_writers = []
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        if self.episode[0]["observation.images.wrist"] is not None:
            output_path_wrist = os.path.join(self.episode_path, 'wrist.mp4')
            video_writer_wrist = VideoWriter(output_path_wrist, fourcc, self.fps, (width, height))
            video_writer_wrist.type = 'wrist'
            video_writers.append(video_writer_wrist)

        # if self.episode[0]["observation.images.exo"] is not None:
        #     output_path_exo = os.path.join(self.episode_path, 'exo.mp4')
        #     video_writer_exo = VideoWriter(output_path_exo, fourcc, self.fps, (width, height))
        #     video_writer_exo.type = 'exo'
        #     video_writers.append(video_writer_exo)

        if self.episode[0]["observation.images.table"] is not None:
            output_path_table = os.path.join(self.episode_path, 'table.mp4')
            video_writer_table = VideoWriter(output_path_table, fourcc, self.fps, (width, height))
            video_writer_table.type = 'table'
            video_writers.append(video_writer_table)

        for item in self.episode:
            for video_writer in video_writers:
                image = item[f'observation.images.{video_writer.type}']
                video_writer.write(image)

        for video_writer in video_writers:
            video_writer.release()
