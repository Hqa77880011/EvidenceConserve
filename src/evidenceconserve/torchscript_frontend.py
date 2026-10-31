"""JSON capture adapter for an already-trained local visual evidence model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imagenet-normalize", action="store_true")
    args = parser.parse_args()
    request = json.load(sys.stdin)
    if request.get("protocol") != "evidenceconserve.capture.v1":
        raise ValueError("Unsupported capture protocol")
    with Image.open(request["view"]["image"]) as source:
        image = np.asarray(source.convert("RGB").resize(
            (args.image_size, args.image_size), Image.Resampling.BILINEAR
        ), dtype=np.float32) / 255
    tensor = torch.from_numpy(image.transpose(2, 0, 1).copy())[None].to(args.device)
    if args.imagenet_normalize:
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tensor = (tensor - mean) / std
    with Path(args.checkpoint).open("rb") as stream:
        model = torch.jit.load(stream, map_location=args.device).eval()
    with torch.inference_mode():
        output = model(tensor)
    evidence = output["evidence"] if isinstance(output, dict) else output
    response = {"evidence": evidence.detach().cpu().reshape(-1).tolist()}
    if isinstance(output, dict) and "quality" in output:
        response["quality"] = float(output["quality"].detach().cpu().item())
    print(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
