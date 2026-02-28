# WandBで探索を行うスクリプトです
# 探索内容：オプティマイザ、1epoch毎の、学習率(縮小スケール値float)、スパース率(ターゲット〜1.0)の組み合わせを探索します
# 例：1epoch目：学習率0.001、スパース率0.5、2epoch目：学習率0.0001、スパース率0.8、など
# オプティマイザは、SGDとAdamを探索します
import argparse

import wandb

import torch
import torchvision
from torch.nn.utils import prune as torch_prune
from tqdm import tqdm

# ImageNet 1000クラス分類前提の学習スクリプトおよびユーティリティ
# 他のスクリプトから呼び出されるベーススクリプト


def get_model(name: str, pretrained: bool = True) -> torch.nn.Module:
    if name == "resnet18":
        return torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
    if name == "resnet50":
        return torchvision.models.resnet50(weights="DEFAULT" if pretrained else None)
    raise ValueError(f"Unsupported model name: {name}, supported: resnet18, resnet50")


def get_optimizer(name: str, model: torch.nn.Module, lr: float = 0.1) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    raise ValueError(f"Unsupported optimizer name: {name}, supported: sgd, adam")


def get_dataset(name: str) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if name != "imagenet":
        raise ValueError(f"Unsupported dataset name: {name}, supported: imagenet")

    train_dir = "/tools/datasets/imagenet/train"
    val_dir = "/tools/datasets/imagenet/val"

    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224), torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_set = torchvision.datasets.ImageFolder(root=val_dir, transform=val_transform)
    return train_set, val_set


def get_dataloader(dataset: torch.utils.data.Dataset, shuffle: bool = True, num_workers: int = 8) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, drop_last=shuffle)


def get_cuda_device(gpu_index: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    gpu_count = torch.cuda.device_count()
    if gpu_index < 0 or gpu_index >= gpu_count:
        raise ValueError(f"--gpu must be in [0, {gpu_count - 1}], got {gpu_index}")
    torch.cuda.set_device(gpu_index)
    return torch.device(f"cuda:{gpu_index}")


def prune_model(model: torch.nn.Module, sparsity: float) -> torch.nn.Module:
    target_model = model
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0.0, 1.0], got {sparsity}")
    parameters_to_prune: list[tuple[torch.nn.Module, str]] = []
    for module in target_model.modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        if module.in_channels <= 3:
            continue
        if module.groups == module.in_channels:
            continue
        parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        return model

    torch_prune.global_unstructured(parameters_to_prune, pruning_method=torch_prune.L1Unstructured, amount=sparsity)
    for module, param_name in parameters_to_prune:
        torch_prune.remove(module, param_name)

    return model

def train(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0

    num_batches = 0
    for inputs, labels in tqdm(dataloader, desc="train", leave=False):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda"):
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        accuracy = (preds == labels).float().mean().item()
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
    if num_batches == 0:
        raise RuntimeError("No training batches were processed. Check dataset or batch limits.")
    return total_loss / num_batches, total_accuracy / num_batches


def val(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0

    with torch.no_grad():
        num_batches = 0
        for inputs, labels in tqdm(dataloader, desc="val", leave=False):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda"):
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == labels).float().mean().item()
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
    if num_batches == 0:
        raise RuntimeError("No validation batches were processed. Check dataset or batch limits.")
    return total_loss / num_batches, total_accuracy / num_batches

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a W&B sweep for ImageNet training.")
    parser.add_argument("--sp", type=float, default=0.85, help="Final-epoch sparsity")
    parser.add_argument("--dataset", type=str, default="imagenet", help="Dataset name")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model weights")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per run")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use (e.g. --gpu 0, --gpu 1)")
    parser.add_argument("--project", type=str, default="imagenet-sweep", help="W&B project name")
    parser.add_argument("--sweep_id", type=str, default=None, help="Existing sweep id to attach agent")
    parser.add_argument("--count", type=int, default=100, help="Number of sweep runs")
    return parser.parse_args()


def build_sweep_config(epochs: int, sp: float) -> dict:
    parameters = {
        "optimizer": {"values": ["sgd", "adam"]}
    }
    for epoch_idx in range(epochs):
        parameters[f"lr{epoch_idx}"] = {"distribution": "log_uniform_values", "min": 1e-6, "max": 1.0}
        parameters[f"sp{epoch_idx}"] = {"distribution": "q_uniform", "min": 0.0, "max": sp, "q": 0.01}

    parameters[f"sp{epochs - 1}"] = {"values": [sp]}

    return {
        "method": "bayes",
        "metric": {"name": "val/top1", "goal": "maximize"},
        "parameters": parameters,
    }


def run_one(args: argparse.Namespace, device) -> None:
    with wandb.init(project=args.project) as run:
        cfg = run.config

        model = get_model(args.model, pretrained=args.pretrained)
        model = model.to(device)
        optimizer = get_optimizer(str(cfg.optimizer), model, lr=float(cfg["lr0"]))

        train_set, val_set = get_dataset(args.dataset)
        train_loader = get_dataloader(train_set, shuffle=True, num_workers=args.num_workers)
        val_loader = get_dataloader(val_set, shuffle=False, num_workers=args.num_workers)

        for epoch in range(args.epochs):
            current_lr = float(cfg[f"lr{epoch}"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr
            model = prune_model(model, float(cfg[f"sp{epoch}"]))

            train_loss, train_acc = train(model, train_loader, optimizer, device)
            val_loss, val_acc = val(model, val_loader, device)

            wandb.log({"epoch": epoch, "train/loss": train_loss, "train/top1": train_acc, "val/loss": val_loss, "val/top1": val_acc})


if __name__ == '__main__':
    args = get_args()
    device = get_cuda_device(args.gpu)
    print(f"Using device: {device}")

    sweep_id = args.sweep_id
    if sweep_id is None:
        sweep_id = wandb.sweep(build_sweep_config(args.epochs, args.sp), project=args.project)
    wandb.agent(sweep_id, function=lambda: run_one(args, device), count=args.count, project=args.project)
