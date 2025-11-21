import os


def start_learn(
    trainer,
    optimizer,
    train_loader,
    test_loader,
    train_dataset,
    scheduler,
    project_root,
    config,
):
    best_acc = 0

    epoch_acc_counter = []

    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print("-" * 50)

        # Обучение
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch + 1)

        # Валидация
        val_loss, val_acc, predictions, targets, probabilities = trainer.validate(
            test_loader, epoch + 1
        )

        # if epoch >= 15:
        #     epoch_acc_counter.append(val_acc)
        #     if (epoch >= 25) and (
        #         (
        #             abs(epoch_acc_counter[0] - epoch_acc_counter[-1])
        #             <= sum(epoch_acc_counter) / len(epoch_acc_counter)
        #         )
        #         or (
        #             any([acc for acc in epoch_acc_counter >= 0.89])
        #             and (
        #                 (sum(epoch_acc_counter) / len(epoch_acc_counter))
        #                 <= val_acc - 0.01
        #                 or sum(epoch_acc_counter) / len(epoch_acc_counter)
        #             )
        #             <= val_acc + 0.01
        #         )
        #     ):
        #         optimizer.param_groups[0]["lr"] = 0.001

        trainer.update_history(train_loss, train_acc, val_loss, val_acc, epoch + 1)

        mean_ap = trainer.log_pr_curves(
            probabilities, targets, train_dataset.classes, epoch + 1
        )
        trainer.log_confusion_matrix(
            predictions, targets, train_dataset.classes, epoch + 1
        )

        # Learning rate scheduling
        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f}\t Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}\t Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Mean AP: {mean_ap:.4f}")

        # Сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(
                project_root, "outputs", "models", "best_model.pth"
            )
            trainer.save_checkpoint(checkpoint_path, epoch + 1, val_acc)
            print(f"Best model saved. Accuracy: {val_acc:.2f}%")

    trainer.close()
    print(f"\nTraining completed. Best accuracy: {best_acc:.2f}%")
