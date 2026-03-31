from pathlib import Path
import torch
import timm

def to_onnx(model_name: str, opset_version: int, output_dir: Path, cache: bool = True) -> None:
    output_path = output_dir / "model.onnx"
    if cache and output_path.exists():
        return

    # model_name expected to be like "hf-hub:BVRA/MegaDescriptor-L-384"
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    dummy_input = torch.randn(1, 3, 384, 384)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
