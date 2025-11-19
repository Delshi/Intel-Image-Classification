import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import io
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        config,
        log_dir="outputs/logs",
        use_tensorboard=True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logs: {log_dir}")

        # История обучения
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        # Счетчик шагов для логирования градиентов
        self.global_step = 0

        self.model.to(device)

    def train_epoch(self, dataloader, epoch):
        """Одна эпоха обучения"""
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (x, y) in enumerate(
            tqdm(dataloader, desc=f"Training Epoch {epoch}")
        ):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            # Логируем градиенты и веса каждые 100 батчей
            if self.use_tensorboard and batch_idx % 100 == 0:
                self._log_gradients_and_weights(epoch, batch_idx)

            self.global_step += 1

        # Основные метрики
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(dataloader)

        # Логируем loss и accuracy
        if self.use_tensorboard:
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/train", accuracy, epoch)
            self.writer.add_scalar(
                "Learning_rate", self.optimizer.param_groups[0]["lr"], epoch
            )

        return avg_loss, accuracy

    def _log_gradients_and_weights(self, epoch, batch_idx):
        """Логирует градиенты и веса с перспективой"""
        step = self.global_step

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Weights
                if "weight" in name and param.data is not None:
                    self.writer.add_histogram(f"weights/{name}", param.data, step)
                    self.writer.add_scalar(
                        f"weights/{name}_mean", param.data.mean(), step
                    )

                # Biases
                elif "bias" in name and param.data is not None:
                    self.writer.add_histogram(f"biases/{name}", param.data, step)
                    self.writer.add_scalar(
                        f"biases/{name}_mean", param.data.mean(), step
                    )

                # Gradients
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, step)
                    self.writer.add_scalar(
                        f"gradients/{name}_mean", param.grad.mean(), step
                    )

    def validate(self, dataloader, epoch):
        """Валидация модели"""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_predictions, all_targets, all_probabilities = [], [], []

        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Validation"):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                all_probabilities.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(dataloader)

        # Логируем validation метрики
        if self.use_tensorboard:
            self.writer.add_scalar("Loss/test", avg_loss, epoch)
            self.writer.add_scalar("Accuracy/test", accuracy, epoch)

        return avg_loss, accuracy, all_predictions, all_targets, all_probabilities

    def log_pr_curves(self, probabilities, targets, classes, epoch):
        """Логирует Precision-Recall curves"""
        if not self.use_tensorboard:
            return 0

        try:
            probs = np.array(probabilities)
            true_labels = np.array(targets)
            n_classes = len(classes)

            # Для многоклассовой классификации
            true_labels_bin = label_binarize(true_labels, classes=range(n_classes))

            # Логируем PR-кривые для каждого класса
            for i, class_name in enumerate(classes):
                precision, recall, _ = precision_recall_curve(
                    true_labels_bin[:, i], probs[:, i]
                )
                ap_score = average_precision_score(true_labels_bin[:, i], probs[:, i])

                # Создаем PR-кривую как изображение для TensorBoard
                fig = plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label=f"AP={ap_score:.4f}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title(f"PR Curve - {class_name} (Epoch {epoch})")
                plt.legend()
                plt.grid(True)

                # Конвертируем в изображение
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                image = Image.open(buf)
                image_array = np.array(image)

                self.writer.add_image(
                    f"PR_Curves/{class_name}", image_array, epoch, dataformats="HWC"
                )
                plt.close(fig)

                self.writer.add_scalar(f"PR_Curves/AP_{class_name}", ap_score, epoch)

            mean_ap = average_precision_score(true_labels_bin, probs, average="macro")
            self.writer.add_scalar("PR_Curves/Mean_AP", mean_ap, epoch)

            return mean_ap
        except Exception as e:
            print(f"Error in PR curves: {e}")
            return 0

    def log_confusion_matrix(self, predictions, targets, classes, epoch):
        """Логирует матрицу ошибок с нормализацией"""
        if not self.use_tensorboard:
            return

        try:
            cm = confusion_matrix(targets, predictions)

            # Нормализуем матрицу
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)

            # Создаем изображение
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # Ненормализованная матрица
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
                ax=ax1,
            )
            ax1.set_title(f"Confusion Matrix - Epoch {epoch}")
            ax1.set_xlabel("Predicted")
            ax1.set_ylabel("Actual")

            # Нормализованная матрица
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
                ax=ax2,
            )
            ax2.set_title(f"Normalized Confusion Matrix - Epoch {epoch}")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")

            plt.tight_layout()

            # Конвертируем в изображение
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)

            self.writer.add_image(
                "Confusion_Matrix", image_array, epoch, dataformats="HWC"
            )
            plt.close(fig)
        except Exception as e:
            print(f"Error in confusion matrix: {e}")

    def update_history(self, train_loss, train_acc, val_loss, val_acc, epoch):
        """Обновляет историю обучения"""
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        # Логируем историю обучения как графики
        if self.use_tensorboard:
            # Loss history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            ax1.plot(
                range(1, epoch + 1), self.history["train_loss"], label="Train Loss"
            )
            ax1.plot(range(1, epoch + 1), self.history["val_loss"], label="Test Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.set_title("Training and Test Loss")
            ax1.legend()
            ax1.grid(True)

            ax2.plot(
                range(1, epoch + 1), self.history["train_acc"], label="Train Accuracy"
            )
            ax2.plot(
                range(1, epoch + 1), self.history["val_acc"], label="Test Accuracy"
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy (%)")
            ax2.set_title("Training and Test Accuracy")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            image = Image.open(buf)
            image_array = np.array(image)

            self.writer.add_image(
                "Training_History", image_array, epoch, dataformats="HWC"
            )
            plt.close(fig)

    def save_checkpoint(self, path, epoch, accuracy):
        """Сохраняет чекпоинт модели"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "accuracy": accuracy,
                "history": self.history,
            },
            path,
        )
        print(f"Checkpoint saved: {path}")

    def close(self):
        """Закрывает writer TensorBoard"""
        if self.use_tensorboard:
            self.writer.close()
