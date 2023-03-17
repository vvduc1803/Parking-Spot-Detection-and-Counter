# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import all necessary package"""
import config
import torch.nn as nn

from torch.optim import Adam
from dataset import Dataset_Load
from utils import train_step, val_step, load_checkpoint, save_checkpoint
from model import Model

def main():
    # Setup data
    train_loader, val_loader, test_loader, class_labels = Dataset_Load(
        config.TRAIN_DIR,
        config.VAL_DIR,
        config.TEST_DIR,
        train_transform=config.train_transform,
        test_transform=config.test_val_transform,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKER)

    # Setup all necessary of model
    model_yolo = Model().to(config.DEVICE)
    optimizer = Adam(model_yolo.parameters(), lr=config.LR, betas=(0.9, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    # Load check point
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_FILE, model_yolo, optimizer, config.LR, config.DEVICE)

    for epoch in range(config.EPOCHS):

        # Training process
        train_loss, train_acc = train_step(model_yolo, train_loader, loss_fn, optimizer, config.DEVICE)

        # Validation process
        val_loss, val_acc = val_step(model_yolo, val_loader, loss_fn, config.DEVICE)

        # Print information
        print(f'Epoch: {epoch+1}')
        print(f'Train loss: {train_loss:.4f} | Train accuracy: {train_acc * 100:.2f}%')
        print(f'Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc * 100:.2f}%')

        # Sve model each 5 epochs
        if (epoch + 1) % 5 == 0 & config.SAVE_MODEL:
            save_checkpoint(model_yolo, optimizer, "model_checkpoint.pt")
        print('--------------------------------------------------------------------')

    # Final testing process
    test_loss, test_acc = val_step(model_yolo, test_loader, loss_fn, config.DEVICE)
    print(f'Test loss: {test_loss:.4f} | Test accuracy: {test_acc * 100:.2f}%')

if __name__ == '__main__':
    main()


