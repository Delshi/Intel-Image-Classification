import gc
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

import torch  # noqa: E402
import yaml  # noqa: E402
from learn_loop import start_learn  # noqa: E402
from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchviz import make_dot  # noqa: E402

from src.data.datasets import ImageFolderDataset, get_transforms  # noqa: E402
from src.models.cnn import CNN  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402


def main():
    # Загрузка конфига
    config_path = os.path.join(project_root, "config", "training.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Датасет
    train_dir = os.path.join(project_root, config["data"]["train_dir"])
    test_dir = os.path.join(project_root, config["data"]["test_dir"])

    train_dataset = ImageFolderDataset(
        train_dir, transform=get_transforms(config["data"]["img_size"], is_train=True)
    )
    test_dataset = ImageFolderDataset(
        test_dir, transform=get_transforms(config["data"]["img_size"], is_train=False)
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["data"]["batch_size"], shuffle=False
    )

    # Модель и оптимизатор
    model = CNN(num_classes=len(train_dataset.classes)).to(device)
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=config["training"]["learning_rate"],
    #     momentum=config["training"]["momentum"],
    #     weight_decay=config["training"]["weight_decay"],
    #     nesterov=True,
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.000375,
        betas=(0.915, 0.985),
        weight_decay=0.0017,
        amsgrad=True,
    )
    criterion = torch.nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    # dummy_tensor = torch.randn(2, 3, 150, 150).to(device)
    # output = model(dummy_tensor)
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.format = "png"
    # dot.render("cnn_graph")

    # Тренер
    log_dir = os.path.join(project_root, "outputs", "logs")
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        config,
        log_dir=log_dir,
        use_tensorboard=True,
    )

    os.makedirs(os.path.join(project_root, "outputs", "models"), exist_ok=True)

    # Обучение
    start_learn(
        trainer=trainer,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        train_dataset=train_dataset,
        scheduler=scheduler,
        project_root=project_root,
        config=config,
    )

    # del model
    # del optimizer
    # gc.collect()
    # torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
