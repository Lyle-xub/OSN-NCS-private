import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import logging
import matplotlib.pyplot as plt
from optical_network import DDNN


class OpticalNetworkTrainer:
    def __init__(
        self,
        data_path="./data",
        checkpoint_path="/home/ubuntu/control_gui/DANTE/data/000.pth",
        learning_rate=0.01,
        batch_size=20,
        epochs=200,
    ):
        self.data_path = data_path
        self.checkpoint_path = checkpoint_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._prepare_data()
        self._initialize_model()

    def _prepare_data(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((200, 200)),
                transforms.Pad([50, 50], fill=(0), padding_mode="constant"),
            ]
        )
        self.train_dataset = torchvision.datasets.MNIST(
            self.data_path, train=True, transform=transform, download=True
        )
        self.val_dataset = torchvision.datasets.MNIST(
            self.data_path, train=False, transform=transform, download=True
        )
        self.trainloader = DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.testloader = DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def _initialize_model(self):
        self.model = DDNN(300, 200, 1.7e-5, 14.5e-2, 5.32e-7)
        self.model.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.2, patience=20
        )

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint)

    def train(self):
        self._setup_logging()
        best_test_acc = 0
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            test_acc = self._evaluate(epoch)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(
                    self.model.state_dict(),
                    f"./data/{best_test_acc:.2f}_{epoch:03d}.pth",
                )
            logging.info(f"Best test accuracy so far: {best_test_acc:.2f}%")
        logging.info("Finished Training")

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for i, data in enumerate(self.trainloader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs.squeeze(1))
            loss = self.loss_func(outputs, labels)
            loss.backward()
            self.optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.reshape(predicted.shape)).sum().item()
            running_loss += loss.item()
            if i % 25 == 24:
                logging.info(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 25:.3f}")
                running_loss = 0.0
        train_acc = 100 * correct / total
        logging.info(f"Epoch {epoch}: Training accuracy: {train_acc:.2f}%")

    def _evaluate(self, epoch):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(images.squeeze(1))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.reshape(predicted.shape)).sum().item()
        test_acc = 100 * correct / total
        logging.info(f"Epoch {epoch}: Test accuracy: {test_acc:.2f}%")
        return test_acc

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
        )


# trainer = OpticalNetworkTrainer()
# trainer.train()
