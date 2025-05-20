import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm.notebook import tqdm_notebook

from tqdm import tqdm

class Trainer:
    def __init__(self,
                 max_epochs,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 device,
                 train_loader,
                 valid_loader,
                 test_loader,
				 log_every_n_steps=50
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps


    def train_step(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            #### YOUR CODE STARTS HERE ####
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            #### YOUR CODE ENDS HERE ####

            running_loss += loss.item()

            #### YOUR CODE STARTS HERE ####
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            if batch_idx % self.log_every_n_steps == 0:
                avg_loss = running_loss / self.log_every_n_steps
                accuracy = 100 * correct / total
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.2f}%")
            #### YOUR CODE ENDS HERE ####

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.2f}%")
        return avg_loss, accuracy


    def valid_step(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc="Validation")

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                #### YOUR CODE STARTS HERE ####
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                #### YOUR CODE ENDS HERE ####

                running_loss += loss.item()

                #### YOUR CODE STARTS HERE ####
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

                if batch_idx % self.log_every_n_steps == 0:
                    avg_loss = running_loss/self.log_every_n_steps
                    accuracy = 100 * correct / total
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.2f}%")
                #### YOUR CODE ENDS HERE ####

            avg_loss = running_loss / len(self.valid_loader)
            accuracy = 100 * correct / total
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.2f}%")

        return avg_loss, accuracy

    def train(self):
        self.model.to(self.device)
        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []

        for epoch in range(self.max_epochs):
            train_loss, train_acc = self.train_step()
            valid_loss, valid_acc = self.valid_step()

            self.scheduler.step()

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_acc)

            print(f"Epoch {epoch+1}/{self.max_epochs}: train_loss: {train_loss:.4f}, train_acc: {train_acc:.2f}%, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.2f}%")
            print()

        return train_losses, train_accuracies, valid_losses, valid_accuracies

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")

            for inputs, targets in pbar:
                #### YOUR CODE STARTS HERE ####
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                #### YOUR CODE ENDS HERE ####

            accuracy = 100 * correct / total
            print(f"test_acc: {accuracy:.2f}%")

        return accuracy

    def plot_metrics(self, train_losses, train_accuracies, valid_losses, valid_accuracies):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(train_losses, label="train_loss", color="blue")
        ax[0].plot(valid_losses, label="valid_loss", linestyle="--", color="orange")
        ax[0].set_title("Losses")

        ax[1].plot(train_accuracies, label="train_acc", color="blue")
        ax[1].plot(valid_accuracies, label="valid_acc", linestyle="--", color="orange")
        ax[1].set_title("Accuracies")

        for i in range(2):
            ax[i].legend()
            ax[i].grid(0.35)
            ax[i].set_xlabel("Epochs")

        plt.show()