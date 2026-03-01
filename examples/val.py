"""Evaluate ImageNet with optional post-training compression (no retraining)."""

from __future__ import annotations

import argparse
import os
from fnmatch import fnmatch
from collections.abc import Callable, Iterable
from typing import Any

import torch
import torchvision

from cimm.compression import (
    Compression,
    QuantizeFakeActivations,
    QuantizeFakeWeights,
    SparseUnstructuredActivations,
    SparseUnstructuredWeights,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImageNetDataModule:
    def __init__(self, data_dir: str, batch_size: int, num_workers: int) -> None:
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set: torchvision.datasets.ImageFolder | None = None
        self.val_set: torchvision.datasets.ImageFolder | None = None

    def setup(self, stage: str | None = None) -> None:
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        val_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        if stage in (None, "fit"):
            self.train_set = torchvision.datasets.ImageFolder(
                root=f"{self.data_dir}/train",
                transform=train_transform,
            )
            self.val_set = torchvision.datasets.ImageFolder(
                root=f"{self.data_dir}/val",
                transform=val_transform,
            )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.val_set is None:
            raise RuntimeError("Data module is not initialized. Call setup first.")
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )


def build_model(name: str, pretrained: bool) -> torch.nn.Module:
    if name == "resnet18":
        return torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
    if name == "resnet50":
        return torchvision.models.resnet50(weights="DEFAULT" if pretrained else None)
    raise ValueError(f"Unsupported model: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ImageNet with compression only (no retraining)")
    parser.add_argument("--data_dir", type=str, default="/tools/datasets/imagenet", help="ImageNet root containing train/ and val/")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained torchvision weights")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path to load before evaluation")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU index")
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["16-mixed", "32-true"])

    parser.add_argument("--weight_sparsity", type=float, default=None, help="Enable unstructured weight sparsity with this ratio [0,1]")
    parser.add_argument("--activation_sparsity", type=float, default=None, help="Enable unstructured activation sparsity with this ratio [0,1]")
    parser.add_argument("--weight_bits", type=int, default=None, help="Enable fake weight quantization with this bit width")
    parser.add_argument("--activation_bits", type=int, default=None, help="Enable fake activation quantization with this bit width")

    parser.add_argument("--min_elements", type=int, default=64, help="Minimum tensor elements for sparsity ops")
    parser.add_argument("--eps", type=float, default=1e-8, help="Quantization epsilon")
    parser.add_argument(
        "--apply_patterns",
        nargs="*",
        default=["*"],
        help="Glob patterns for module names to compress (e.g. layer*.conv*)",
    )
    parser.add_argument(
        "--apply_module_types",
        nargs="*",
        default=["conv2d", "linear"],
        choices=["conv2d", "linear"],
        help="Module types to compress",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num_workers must be >= 0, got {args.num_workers}")

    if args.weight_sparsity is not None and not 0.0 <= args.weight_sparsity <= 1.0:
        raise ValueError(f"--weight_sparsity must be in [0, 1], got {args.weight_sparsity}")
    if args.activation_sparsity is not None and not 0.0 <= args.activation_sparsity <= 1.0:
        raise ValueError(f"--activation_sparsity must be in [0, 1], got {args.activation_sparsity}")
    if args.weight_bits is not None and args.weight_bits < 2:
        raise ValueError(f"--weight_bits must be >= 2, got {args.weight_bits}")
    if args.activation_bits is not None and args.activation_bits < 2:
        raise ValueError(f"--activation_bits must be >= 2, got {args.activation_bits}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    gpu_count = torch.cuda.device_count()
    if args.gpu < 0 or args.gpu >= gpu_count:
        raise ValueError(f"--gpu must be in [0, {gpu_count - 1}], got {args.gpu}")

    if args.checkpoint is not None and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")


def _strip_common_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    transformed = dict(state_dict)

    if all(k.startswith("module.") for k in keys):
        transformed = {k[len("module.") :]: v for k, v in transformed.items()}
        keys = list(transformed.keys())
    if all(k.startswith("model.") for k in keys):
        transformed = {k[len("model.") :]: v for k, v in transformed.items()}

    return transformed


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            state_dict = checkpoint["model_state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format: could not find state dict")
    else:
        raise ValueError("Unsupported checkpoint format: expected dict")

    state_dict = _strip_common_prefixes(state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)

    if load_result.missing_keys:
        print(f"[checkpoint] missing_keys={len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"[checkpoint] unexpected_keys={len(load_result.unexpected_keys)}")


def build_layer_selector(
    wrapped_model: Compression,
    patterns: Iterable[str],
    module_type_names: Iterable[str],
) -> Callable[[torch.nn.Module], bool]:
    module_types: list[type[torch.nn.Module]] = []
    for type_name in module_type_names:
        if type_name == "conv2d":
            module_types.append(torch.nn.Conv2d)
        elif type_name == "linear":
            module_types.append(torch.nn.Linear)

    patterns_tuple = tuple(patterns)
    module_ids: set[int] = set()
    for name, module in wrapped_model.model.named_modules():
        if not name:
            continue
        by_name = any(fnmatch(name, pattern) for pattern in patterns_tuple)
        by_type = bool(module_types) and isinstance(module, tuple(module_types))
        if by_name or by_type:
            module_ids.add(id(module))

    return lambda module: id(module) in module_ids


def apply_compressions(model: Compression, args: argparse.Namespace) -> None:
    layer_selector = build_layer_selector(model, args.apply_patterns, args.apply_module_types)

    if args.weight_sparsity is not None:
        model.attach(
            SparseUnstructuredWeights,
            apply_layers=layer_selector,
            sparsity=float(args.weight_sparsity),
            min_elements=int(args.min_elements),
            mask=True,
        )
    if args.activation_sparsity is not None:
        model.attach(
            SparseUnstructuredActivations,
            apply_layers=layer_selector,
            sparsity=float(args.activation_sparsity),
            min_elements=int(args.min_elements),
            mask=True,
        )
    if args.weight_bits is not None:
        model.attach(
            QuantizeFakeWeights,
            apply_layers=layer_selector,
            bits=int(args.weight_bits),
            eps=float(args.eps),
        )
    if args.activation_bits is not None:
        model.attach(
            QuantizeFakeActivations,
            apply_layers=layer_selector,
            bits=int(args.activation_bits),
            eps=float(args.eps),
        )


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            batch_size = y.size(0)
            total_samples += batch_size
            total_loss += float(loss.detach()) * batch_size

            top1 = logits.argmax(dim=1)
            top1_correct += int((top1 == y).sum().item())

            k = min(5, logits.size(1))
            top5 = torch.topk(logits, k=k, dim=1).indices
            top5_correct += int((top5 == y.unsqueeze(1)).any(dim=1).sum().item())

    avg_loss = total_loss / max(total_samples, 1)
    top1_acc = top1_correct / max(total_samples, 1)
    top5_acc = top5_correct / max(total_samples, 1)
    return avg_loss, top1_acc, top5_acc


def main() -> None:
    args = parse_args()
    validate_args(args)

    device = torch.device(f"cuda:{args.gpu}")
    use_amp = args.precision == "16-mixed"

    model = build_model(args.model, args.pretrained)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, device)

    compressed_model = Compression(model.to(device))
    apply_compressions(compressed_model, args)

    datamodule = ImageNetDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()

    criterion = torch.nn.CrossEntropyLoss().to(device)
    val_loss, val_top1, val_top5 = evaluate(
        model=compressed_model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    print("[eval] no retraining, compression-only validation")
    print(f"[eval] model={args.model} pretrained={args.pretrained} checkpoint={args.checkpoint}")
    print(
        "[eval] compression "
        f"weight_sparsity={args.weight_sparsity} "
        f"activation_sparsity={args.activation_sparsity} "
        f"weight_bits={args.weight_bits} "
        f"activation_bits={args.activation_bits}"
    )
    print(f"[val] loss={val_loss:.4f} top1={val_top1:.4f} top5={val_top5:.4f}")


if __name__ == "__main__":
    main()
