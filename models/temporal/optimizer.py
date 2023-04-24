"""Optimizer to train the baseline model."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.utils as utils

class EarlyStopper:
    """ Early Stopper Class.
    
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class Optimizer:
    """Optimizer Class.

    Attributes
    ----------
    model: torch.nn.Module
        Pytorch model to optimize (e.g. RNNModel, LSTMModel, GRUModel).
    loss_fn: function
        Loss function.
    optimizer: torch.optim
        Optimization algorithm (e.g. Adam).
    reweight_positive: float
        value for loss to upweight positive examples.
    earlystopper: EarlyStopper
        For stopping training early to avoid overfitting.

    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        activation,
        lr_scheduler,
        reweight_positive=None,
        earlystopper=None,
        clipping_value=None,
        
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.activation = activation
        self.lr_scheduler = lr_scheduler
        self.train_losses = []
        self.val_losses = []
        self.device = model.device
        self.reweight_positive = reweight_positive
        self.earlystopper = earlystopper
        self.clipping_value = clipping_value

    def reweight_loss(self, loss, y):
        """Reweight the losses."""
        if isinstance(self.reweight_positive, (float, np.float64)):
            loss[y.squeeze() == 1] *= self.reweight_positive
        elif self.reweight_positive == "mini-batch":
            loss[y.squeeze() == 1] *= (y == 0).sum() / (y == 1).sum()

        return loss

    def train_step(self, x, y):
        """Train model for one step."""
        # Sets model to train mode
        self.model.train(True)

   #     torch.autograd.set_detect_anomaly(True)
        
        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(yhat.squeeze(), y.squeeze())

        # Reweight the losses
        loss = self.reweight_loss(loss, y)

        # Mask out the loss for -1 labels.
        loss *= ~y.eq(-1).squeeze()

        # Take mean of loss.
        loss = loss.sum() / (~y.eq(-1)).sum()

        # Computes gradients
        loss.backward()

#        if self.clipping_value is not None:
#            utils.clip_grad_norm_(self.model.parameters(), self.clipping_value)
        
        # self.model.float()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(
        self,
        train_loader,
        val_loader,
        n_epochs=50,
        model_path=None,
    ):
        """Train pytorch model.

        Parameters
        ----------
        train_loader: DataLoader
            Dataset object containing training set.
        val_loader: DataLoader
            Dataset object containing validation set.
        n_epochs: int
            Number of complete passes through the training set.
        """
        if model_path is None:
            model_path = f'checkpoint_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self.train_step(x_batch, y_batch)

                if not np.isnan(loss).any():
                    batch_losses.append(loss)
                        
                assert not np.isnan(loss).any()

                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(yhat.squeeze(), y_val.squeeze())

                    # Reweight the losses
                    val_loss = self.reweight_loss(val_loss, y_val)

                    # Mask out the loss for -1 labels.
                    val_loss *= ~y_val.eq(-1).squeeze()

                    # Take mean of loss.
                    val_loss = (val_loss.sum() / (~y_val.eq(-1)).sum()).item()

                    if not np.isnan(loss).any():
                        batch_losses.append(loss)
                        
                    assert not np.isnan(val_loss).any()

                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                
            torch.save(self.model.state_dict(), model_path)
            print(
                f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t \
                Validation loss: {validation_loss:.4f}"
            )
#            if self.earlystopper.early_stop(validation_loss):             
#                break
                
            self.lr_scheduler.step()

    def evaluate(
        self,
        test_loader,
        flatten=True,
    ):
        """Evaluate pytorch model.

        Parameters
        ----------
        test_loader: DataLoader
            Dataset object containing test set.

        """
        with torch.no_grad():
            y_pred_values = []
            y_test_labels = []
            y_pred_labels = []
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                self.model.eval()
                y_hat = self.activation(self.model(x_test))
                y_pred_values.append(y_hat.cpu().detach())
                y_test_labels.append(y_test.cpu().detach())
                y_pred_labels.append(torch.round(y_hat).cpu().detach())

        y_test_labels = np.concatenate(y_test_labels)
        y_pred_labels = np.concatenate(y_pred_labels)
        y_pred_values = np.concatenate(y_pred_values)

        if flatten:
            return (y.flatten() for y in [y_test_labels, y_pred_values, y_pred_labels])

        return y_test_labels, y_pred_values, y_pred_labels

    def plot_losses(self):
        """Plot training and validation losses."""
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()
