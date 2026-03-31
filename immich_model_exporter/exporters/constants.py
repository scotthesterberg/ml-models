from enum import StrEnum
from typing import NamedTuple


class ModelSource(StrEnum):
    INSIGHTFACE = "insightface"
    MCLIP = "mclip"
    OPENCLIP = "openclip"
    YOLOV8 = "yolov8"
    MEGADESCRIPTOR = "megadescriptor"


class ModelTask(StrEnum):
    FACIAL_RECOGNITION = "facial-recognition"
    SEARCH = "clip"
    PET_DETECTION = "pet-detection"
    PET_RECOGNITION = "pet-recognition"


class SourceMetadata(NamedTuple):
    name: str
    link: str
    type: str


SOURCE_TO_METADATA = {
    ModelSource.MCLIP: SourceMetadata("M-CLIP", "https://huggingface.co/M-CLIP", "CLIP"),
    ModelSource.OPENCLIP: SourceMetadata("OpenCLIP", "https://github.com/mlfoundations/open_clip", "CLIP"),
    ModelSource.INSIGHTFACE: SourceMetadata(
        "InsightFace", "https://github.com/deepinsight/insightface/tree/master", "facial recognition"
    ),
    ModelSource.YOLOV8: SourceMetadata("Ultralytics", "https://github.com/ultralytics/ultralytics", "object detection"),
    ModelSource.MEGADESCRIPTOR: SourceMetadata(
        "MegaDescriptor", "https://github.com/wildme/megadescriptor", "animal re-identification"
    ),
}


SOURCE_TO_TASK = {
    ModelSource.MCLIP: ModelTask.SEARCH,
    ModelSource.OPENCLIP: ModelTask.SEARCH,
    ModelSource.INSIGHTFACE: ModelTask.FACIAL_RECOGNITION,
    ModelSource.YOLOV8: ModelTask.PET_DETECTION,
    ModelSource.MEGADESCRIPTOR: ModelTask.PET_RECOGNITION,
}

RKNN_SOCS = ["rk3566", "rk3568", "rk3576", "rk3588"]


# glob to delete old UUID blobs when reuploading models
_uuid_char = "[a-fA-F0-9]"
_uuid_glob = _uuid_char * 8 + "-" + _uuid_char * 4 + "-" + _uuid_char * 4 + "-" + _uuid_char * 4 + "-" + _uuid_char * 12
DELETE_PATTERNS = [
    "**/*onnx*",
    "**/Constant*",
    "**/*.weight",
    "**/*.bias",
    "**/*.proj",
    "**/*in_proj_bias",
    "**/*.npy",
    "**/*.latent",
    "**/*.pos_embed",
    f"**/{_uuid_glob}",
]
