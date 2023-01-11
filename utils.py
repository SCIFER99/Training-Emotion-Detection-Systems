# By: Tim Tarver

# Utils.py file to implement a learning rate scheduler
# for the emotion detection system.

from torch.optim import lr_scheduler
import cv2

# This begins the class to develop the learning rate scheduler

class LRScheduler:

    """Check if the validation loss does not decrease for
       a given number of epochs (patience), then decrease the
       the learning rate by a given factor.
       """

    def __init__(self, optimizer, patience = 5,
                 minimum_learning_rate = 1e-6,
                 factor = 0.5):

        """
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        :returns:  new_lr = old_lr * factor
        
        """
        self.optimizer = optimizer
        self.patience = patience
        self.minimum_learning_rate = minimum_learning_rate
        self.factor = factor
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                           mode = "min",
                                                           patience = self.patience,
                                                           factor = self.factor,
                                                           minimum_learning_rate = self.minimum_learning_rate,
                                                           verbose = True)

        # Now call the validation_loss() function to record that
        # data into the learning rate scheduler

        def __call__(self, validation_loss):

            self.lr_scheduler.step(validation_loss)

# Now we create a new class to stop the training procedure when the loss
# does not improve over a certain number of iterations.

class EarlyStop:

    def __init__(self, patience = 10, minimum_delta = 0):

        """
        :param patience: number of epochs to wait stopping the training procedure
        :param min_delta: the minimum difference between (previous and the new loss)
        to consider the network is improving.
        """

        self.early_stop_enabled = False
        self.minimum_delta = minimum_delta
        self.patience = patience
        self.best_loss = None
        self.counter = 0

    # Now enable the ability to execute the Early Stop class
    # whenever the validation loss as an argument is supplied to the
    # object of the Early Stop class.

    def __call__(self, validation_loss):

        # Update the validation loss if the condition doesn't hold
        if self.best_loss is None:
            self.best_loss = validation_loss

        # Checks if the training procedure should be stopped
        elif (self.best_loss - validation_loss) < self.minimum_delta:
            self.counter += 1
            print(f"[INFO] Early Stop: {self.counter}/{self.patience}... \n\n")

            if self.counter >= self.patience:
                self.early_stop_enabled = True
                print(f"[INFO] Early stopping enabled")

        # Resets the Early Stopping Counter
        elif (self.best_loss - validation_loss) > self.minimum_delta:
            self.best_loss = validation_loss
            self.counter = 0

            
