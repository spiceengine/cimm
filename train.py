"""ImageNet training with plain PyTorch."""

import argparse
import random
from typing import Literal

import torch
from torch.nn.utils import prune as torch_prune

from utils.datasets import ImageNetDataModule
from utils.models import build_model


def build_linear_schedule(start: float, end: float, epochs: int) -> list[float]:
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if epochs == 1:
        return [float(end)]
    return [start + (end - start) * epoch / (epochs - 1) for epoch in range(epochs)]


def prune_conv_weights(model: torch.nn.Module, sparsity: float) -> None:
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0.0, 1.0], got {sparsity}")

    targets: list[tuple[torch.nn.Module, str]] = []
    for module in model.modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        if module.in_channels <= 3:
            continue
        if module.groups == module.in_channels:
            continue
        targets.append((module, "weight"))

    if not targets:
        return

    torch_prune.global_unstructured(
        targets,
        pruning_method=torch_prune.L1Unstructured,
        amount=sparsity,
    )
    for module, param_name in targets:
        torch_prune.remove(module, param_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = y.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
        total_samples += batch_size

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
            total_samples += batch_size

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ImageNet classifier with PyTorch")
    parser.add_argument("--data_dir", type=str, default="/tools/datasets/imagenet", help="ImageNet root containing train/ and val/")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained torchvision weights")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU index")
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["16-mixed", "32-true"])
    parser.add_argument("--lr_start", type=float, default=0.1)
    parser.add_argument("--lr_end", type=float, default=0.001)
    parser.add_argument("--sp_start", type=float, default=0.0)
    parser.add_argument("--sp_end", type=float, default=0.85)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9, help="Used only for SGD")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if not 0.0 <= args.sp_start <= 1.0:
        raise ValueError(f"--sp_start must be in [0, 1], got {args.sp_start}")
    if not 0.0 <= args.sp_end <= 1.0:
        raise ValueError(f"--sp_end must be in [0, 1], got {args.sp_end}")
    if args.batch_size < 1:
        raise ValueError(f"--batch_size must be >= 1, got {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num_workers must be >= 0, got {args.num_workers}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    gpu_count = torch.cuda.device_count()
    if args.gpu < 0 or args.gpu >= gpu_count:
        raise ValueError(f"--gpu must be in [0, {gpu_count - 1}], got {args.gpu}")


def build_optimizer(
    model: torch.nn.Module,
    optimizer_name: Literal["sgd", "adam"],
    initial_lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=initial_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=initial_lr,
        weight_decay=weight_decay,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(f"cuda:{args.gpu}")
    use_amp = args.precision == "16-mixed"

    lr_schedule = build_linear_schedule(args.lr_start, args.lr_end, args.epochs)
    sparsity_schedule = build_linear_schedule(args.sp_start, args.sp_end, args.epochs)

    datamodule = ImageNetDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = build_model(args.model, args.pretrained).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        initial_lr=float(lr_schedule[0]),
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        lr = float(lr_schedule[epoch])
        sparsity = float(sparsity_schedule[epoch])

        for group in optimizer.param_groups:
            group["lr"] = lr

        prune_conv_weights(model, sparsity)

        train_loss, train_top1 = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_loss, val_top1 = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )

        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"lr={lr:.6f} sp={sparsity:.4f} "
            f"train_loss={train_loss:.4f} train_top1={train_top1:.4f} "
            f"val_loss={val_loss:.4f} val_top1={val_top1:.4f}"
        )


if __name__ == "__main__":
    main()
