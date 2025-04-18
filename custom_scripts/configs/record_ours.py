from dataclasses import dataclass

@dataclass
class RecordOursPipelineConfig:
    dataset_path: str
    episode_num: int
    episode_len: int
    create_video: bool = True
    use_devices: bool = True
    task: str | None = None
    fps: int = 30