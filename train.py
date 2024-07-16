#!/usr/bin/env python3

"""Train network using PyTorch

Author: Baiyu
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import (
    get_network,
    get_training_dataloader,
    get_val_dataloader,
    WarmUpLR,
    most_recent_folder,
    most_recent_weights,
    last_epoch,
    best_acc_weights,
)


def train(epoch):
    start = time.time()
    net.train()

    # Initialize variables to calculate average loss
    total_loss = 0.0
    correct = 0
    total_samples = 0

    # Create a tqdm progress bar
    pbar = tqdm(
        total=len(cangjie_training_loader), desc=f"Epoch {epoch} Training", unit="batch"
    )

    for batch_index, (images, labels) in enumerate(cangjie_training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total_samples += labels.size(0)

        n_iter = (epoch - 1) * len(cangjie_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if "weight" in name:
                writer.add_scalar(
                    "LastLayerGradients/grad_norm2_weights", para.grad.norm(), n_iter
                )
            if "bias" in name:
                writer.add_scalar(
                    "LastLayerGradients/grad_norm2_bias", para.grad.norm(), n_iter
                )

        # Update the progress bar
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", lr=f'{optimizer.param_groups[0]["lr"]:.6f}'
        )
        pbar.update(1)

        # Update training loss for each iteration
        writer.add_scalar("Train/loss", loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    pbar.close()

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram(f"{layer}/{attr}", param, epoch)

    finish = time.time()

    print(f"Epoch {epoch} training time consumed: {finish - start:.2f}s")
    print(
        f"Train set: Epoch: {epoch}, Average loss: {avg_loss:.7f}, Accuracy: {accuracy:.7f}"
    )


@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0

    pbar = tqdm(total=len(cangjie_test_loader), desc="Evaluating", unit="batch")

    for images, labels in cangjie_test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        pbar.update(1)

    pbar.close()
    finish = time.time()

    if args.gpu:
        print("GPU INFO.....")
        print(torch.cuda.memory_summary(), end="")

    print("Evaluating Network.....")
    print(
        f"Validation set: Epoch: {epoch}, Average loss: {test_loss / len(cangjie_test_loader.dataset):.7f}, "
        f"Accuracy: {correct.float() / len(cangjie_test_loader.dataset):.7f}, Time consumed: {finish - start:.7f}s"
    )
    print()

    if tb:
        writer.add_scalar(
            "Test/Average loss", test_loss / len(cangjie_test_loader.dataset), epoch
        )
        writer.add_scalar(
            "Test/Accuracy", correct.float() / len(cangjie_test_loader.dataset), epoch
        )

    return correct.float() / len(cangjie_test_loader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="Net type")
    parser.add_argument(
        "-gpu", action="store_true", default=False, help="Use GPU or not"
    )
    parser.add_argument("-b", type=int, default=128, help="Batch size for DataLoader")
    parser.add_argument("-warm", type=int, default=1, help="Warm-up training phase")
    parser.add_argument("-lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument(
        "-resume", action="store_true", default=False, help="Resume training"
    )
    parser.add_argument(
        "-data_dir", type=str, default="./data", help="Specific data location"
    )
    args = parser.parse_args()

    net = get_network(args)

    cangjie_training_loader = get_training_dataloader(
        args.data_dir,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=32,
        batch_size=args.b,
        shuffle=True,
        dataset="cangjie",
    )

    cangjie_test_loader = get_val_dataloader(
        args.data_dir,
        settings.CIFAR100_TEST_MEAN,
        settings.CIFAR100_TEST_STD,
        num_workers=32,
        batch_size=args.b,
        shuffle=True,
        dataset="cangjie",
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings.MILESTONES, gamma=0.2
    )
    iter_per_epoch = len(cangjie_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(
            os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT
        )
        if not recent_folder:
            raise Exception("No recent folder were found")

        checkpoint_path = os.path.join(
            settings.CHECKPOINT_PATH, args.net, recent_folder
        )

    else:
        checkpoint_path = os.path.join(
            settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW
        )

    if not os.path.exists(settings.LOG_DIR):
        os.makedirs(settings.LOG_DIR)

    writer = SummaryWriter(
        log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)
    )
    input_tensor = torch.Tensor(1, 1, 64, 64)  # NCHW
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, "{net}-{epoch}-{type}.pth")

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(
            os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
        )
        if best_weights:
            weights_path = os.path.join(
                settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights
            )
            print("Found best accuracy weights file: {}".format(weights_path))
            print("Load best training file to test accuracy...")
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print("Best accuracy is {:0.2f}".format(best_acc))

        recent_weights_file = most_recent_weights(
            os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
        )
        if not recent_weights_file:
            raise Exception("No recent weights file were found")
        weights_path = os.path.join(
            settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file
        )
        print("Loading weights file {} to resume training.....".format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(
            os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)
        )

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step()  # Updated to remove epoch parameter

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(
                net=args.net, epoch=epoch, type="best"
            )
            print(f"Saving best weights file to {weights_path}")
            torch.save(net.state_dict(), weights_path)
            best_acc = acc

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(
                net=args.net, epoch=epoch, type="regular"
            )
            print(f"Saving regular weights file to {weights_path}")
            torch.save(net.state_dict(), weights_path)

    writer.close()
