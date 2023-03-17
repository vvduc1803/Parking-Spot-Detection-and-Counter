# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import cv2
import torch
import numpy as np
import config
from tqdm.auto import tqdm

def calc_diff(im1, im2):
    # Return difference about mean of 2 image
    return np.abs(np.mean(im1) - np.mean(im2))
def get_parking_spots_bboxes(connected_components):
    # Get parking spots boxes and number of parking spots
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT])
        y1 = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])

        slots.append([x1, y1, w, h])

    return totalLabels, slots

def empty_or_not(model, spot_bgr, device):

    # Convert and resize image
    spot_bgr = np.array(spot_bgr)
    spot_bgr = cv2.resize(spot_bgr, (65, 35))

    # Transform image and convert image, model to device
    model = model.to(device)
    img_resized = config.test_val_transform(spot_bgr).reshape(1, -1).to(device)

    # Compute parking spots empty or not
    results = model(img_resized)
    results = torch.sigmoid(results)
    results = 0 if results < 0.5 else 1

    # Return results
    if results == 0:
        return False
    else:
        return True


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Apply tqdm for visual loading process
    loop = tqdm(dataloader, leave=True)

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(loop):

        # Send data to target device
        X, y = X.to(device), y.type(torch.float32).to(device)

        # 1. Forward pass
        y_pred = model(X)
        y_pred = torch.squeeze(y_pred, dim=1)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.sigmoid(y_pred)
        for i in range(len(y_pred_class)):
            y_pred_class[i] = 0 if y_pred_class[i] < 0.5 else 1
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device):
    """Tests(Validation) a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a validation dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (val_loss, val_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Apply tqdm for visual loading process
    loop = tqdm(dataloader, leave=True)

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(loop):
            # Send data to target device
            X, y = X.to(device), y.type(torch.float32).to(device)

            # 1. Forward pass
            y_pred = model(X)
            y_pred = torch.squeeze(y_pred, dim=1)

            # 2. Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            y_pred_class = torch.sigmoid(y_pred)
            for i in range(len(y_pred_class)):
                y_pred_class[i] = 0 if y_pred_class[i] < 0.5 else 1
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def save_checkpoint(model, optimizer, filename="my_checkpoint.pt"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Meaningful when continue train model, avoid load old checkpoint lead to many hours of debugging
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
