# Necessary Imports
import torch
import torch.optim as optim
from copy import deepcopy
import configparser
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

# Custom Imports
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.torch_utils import de_parallel
from yolov5.utils.general import strip_optimizer
from dataloader import train_dataloader, valid_dataloader


# Read config file
def load_config():
    config = configparser.ConfigParser()
    config.read("config.cfg")

    # Get hyperparameters from config
    lr = float(config["HYPERPARAMETERS"]["learning_rate"])
    num_epochs = int(config["HYPERPARAMETERS"]["num_epochs"])
    device = config["HYPERPARAMETERS"]["device"]

    # Get file paths from config
    yolov5_source_model_path = config["FILE_PATHS"]["yolov5_source_model_path"]
    output_model_path = config["FILE_PATHS"]["output_model_path"]

    return yolov5_source_model_path, output_model_path, lr, num_epochs, device


class Model:
    def __init__(
        self, yolov5_source_model_path, output_model_path, lr, num_epochs, device
    ):
        self.yolov5_source_model_path = yolov5_source_model_path
        self.output_model_path = output_model_path
        self.model = self.load_model()
        self.compute_loss, self.optimizer = self.initialize_loss_fn_and_optimizer(
            self.model
        )
        self.set_hyperparameters(lr, num_epochs, device)
        self.writer = SummaryWriter()

    def load_model(self):
        # Loading Model
        print("Loading Model...")
        model = torch.load(self.yolov5_source_model_path)
        model = model["model"]
        model = model.to(device)

        # Set the requires_grad to True for parameters
        for k, v in model.named_parameters():
            v.requires_grad = True
        return model

    def set_hyperparameters(self, lr, num_epochs, device):
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device

    def initialize_loss_fn_and_optimizer(self, model):
        # Initialize loss fn and optimizers
        print("Initializing loss function and optimizer...")
        compute_loss = ComputeLoss(model)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        return compute_loss, optimizer

    def train(self):
        print(f"Starting Training for {self.num_epochs} epochs!")
        # Train/Valid Pipeline
        for epoch in tqdm(range(self.num_epochs), desc="Epochs"):
            training_loss = 0.0
            # total_train_samples = 0
            self.model.train()

            for images, targets, paths in tqdm(
                train_dataloader, desc="Training Batch:"
            ):
                images = images.to(self.device, non_blocking=False).float() / 255
                images = images.half()
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss, loss_items = self.compute_loss(outputs, targets.to(self.device))
                # loss = torch.autograd.Variable(loss, requires_grad=True)
                # loss.requires_grad=True
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()  # * images.size(0)
                # total_train_samples += images.size(0)

            # training_loss_per_sample = training_loss / total_train_samples
            training_loss = training_loss / len(train_dataloader)

            validation_loss = 0.0
            total_valid_samples = 0
            with torch.no_grad():
                for images, targets, _ in tqdm(
                    valid_dataloader, desc="Validation Batch:"
                ):
                    images = images.to(self.device).float() / 255
                    images = images.half()
                    outputs = self.model(images)
                    loss, loss_items = self.compute_loss(
                        outputs, targets.to(self.device)
                    )
                    validation_loss += loss.item()  # * images.size(0)
                    # total_valid_samples += images.size(0)

            # validation_loss_per_sample = validation_loss / total_valid_samples
            validation_loss = validation_loss / len(valid_dataloader)
            self.log_metrics(training_loss, validation_loss, epoch)
            print(
                f"Epoch [{epoch+1}/{self.num_epochs}, Training Loss: {training_loss:.4f}, Validation Loss:{validation_loss:.4f}]"
            )

            # Append epoch number to the output model path
            base_path, ext = os.path.splitext(self.output_model_path)
            output_model_path = f"{base_path}_epoch_{epoch}{ext}"
            self.save_model(output_model_path)

        print(f"Training Completed for {self.num_epochs} epochs!")
        # Close the tensorboard writer after training

        self.writer.close()
        self.clean_cache()

    def log_metrics(self, training_loss, validation_loss, epoch):
        # Log metrics to tensorboard
        self.writer.add_scalars(
            "Loss", {"train": training_loss, "validation": validation_loss}, epoch
        )
        self.writer.flush()

    def save_model(self, output_model_path):
        print("Saving the model...")
        ckpt = {
            "epoch": None,
            "best_fitness": None,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": None,
            "updates": None,
            "optimizer": None,
            "opt": None,
            "git": None,
            "date": None,
        }
        torch.save(ckpt, output_model_path)
        strip_optimizer(output_model_path)
        print("Model successfully saved!")

    def clean_cache(self):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    yolov5_source_model_path, output_model_path, lr, num_epochs, device = load_config()
    model = Model(yolov5_source_model_path, output_model_path, lr, num_epochs, device)
    model.train()
