#!/usr/bin/env python3

"""Test neural network performance
Print Top-1 and Top-5 error rates on the test dataset of a model.

Author: baiyu
"""

import argparse
from tqdm import tqdm  # Import tqdm for progress bars
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from conf import settings
from utils import get_network, get_val_dataloader


def compute_accuracy(predictions, labels, top_k=1):
    """Compute accuracy for top-k predictions."""
    batch_size = labels.size(0)
    top_k_correct = 0

    # Iterate over each item in the batch
    for i in range(batch_size):
        # Get the top-k predictions for the current item
        top_k_preds = predictions[i, :top_k]
        # Check if the true label is in the top-k predictions
        if labels[i].item() in top_k_preds.cpu().numpy():
            top_k_correct += 1

    return top_k_correct


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", type=str, required=True, help="Net type")
    parser.add_argument(
        "-weights", type=str, required=True, help="The weights file you want to test"
    )
    parser.add_argument(
        "-gpu", action="store_true", default=False, help="Use GPU or not"
    )
    parser.add_argument("-b", type=int, default=32, help="Batch size for dataloader")
    parser.add_argument("-data_dir", type=str, default="./data")
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_val_dataloader(
        args.data_dir,
        settings.CIFAR100_TEST_MEAN,
        settings.CIFAR100_TEST_STD,
        num_workers=32,
        batch_size=args.b,
        dataset="cangjie",
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        # Initialize tqdm progress bar
        pbar = tqdm(total=len(cifar100_test_loader), desc="Evaluating", unit="batch")

        for images, labels in cifar100_test_loader:
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()

            outputs = net(images)
            _, preds = outputs.topk(5, 1, largest=True, sorted=True)

            # Compute top-1 and top-5 accuracy
            top5_correct = compute_accuracy(preds, labels, top_k=5)
            top1_correct = compute_accuracy(preds, labels, top_k=1)

            correct_5 += top5_correct
            correct_1 += top1_correct

            # Update progress bar
            pbar.update(1)

        pbar.close()

    # Compute and print the Top-1 and Top-5 accuracy
    total_samples = len(cifar100_test_loader.dataset)
    top1_accuracy = correct_1 / total_samples
    top5_accuracy = correct_5 / total_samples

    print(f"Top-1 Accuracy: {top1_accuracy:.7f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.7f}")
    print(f"Top-1 Error Rate: {1 - top1_accuracy:.7f}")
    print(f"Top-5 Error Rate: {1 - top5_accuracy:.7f}")
    print(f"Parameter numbers: {sum(p.numel() for p in net.parameters())}")

    if args.gpu:
        print("GPU INFO.....")
        print(torch.cuda.memory_summary(), end="")
