from pathlib import Path
from ultralytics import YOLO

def to_onnx(model_name: str, opset_version: int, output_dir: Path, cache: bool = True) -> None:
    output_path = output_dir / "model.onnx"
    if cache and output_path.exists():
        return

    model = YOLO(model_name)
    model.export(format="onnx", opset=opset_version, simplify=True)
    
    # Ultralytics exports to the same directory as the model file by default
    exported_path = Path(model_name).with_suffix(".onnx")
    output_dir.mkdir(parents=True, exist_ok=True)
    exported_path.replace(output_path)
