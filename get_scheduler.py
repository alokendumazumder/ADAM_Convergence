from torch.optim.lr_scheduler import StepLR, LinearLR, ExponentialLR, OneCycleLR
import numpy as np

class Schedulers:
    def __init__(self, optimizer, initial_lr, name, threshold=2000, const=0.01):
        self.optimizer = optimizer
        self.lr = initial_lr
        self.name = name
        self.threshold = threshold
        self.const = const
        self.num_T = 0
        self.epoch = 0
        self.lr_list = []

        if self.name=="sqrt" and threshold <= 0:
            raise Exception("threshold > 0 for sqrt scheduler")

        assert name in ["sqrt", "ours", "inverse_time_decay", "exponential_decay", "cosine_decay"]

    def sqrt(self, lr):
        return 0.01 / np.sqrt(self.epoch)

    def inverse_time_decay(self, lr):
        return 0.01 / (1 + 0.01*self.epoch)

    def exponential_decay(self, lr):
        return 0.01 * np.exp(-0.01*self.epoch)

    def cosine_decay(self):
        return 0.005 + (0.01 - 0.005)*(1 + np.cos(np.pi * self.epoch / 100)) / 2

    def step(self, const_loss=None, epochs=None, lipshitz=None):
        self.epoch += 1
        if self.name == "sqrt":
            self.lr = self.sqrt(self.lr)
        if self.name == "inverse_time_decay":
            self.lr = self.inverse_time_decay(self.lr)
        if self.name == "cosine_decay":
            self.lr = self.cosine_decay()
        if self.name == "exponential_decay":
            self.lr = self.exponential_decay(self.lr)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
            self.lr_list.append(self.lr)

def get_scheduler(scheduler_name, optimizer, lr, step_num, initial_lr, train_loader_len, epochs):
    if scheduler_name == "ours":
        scheduler = Schedulers(optimizer, lr, scheduler_name)
    elif scheduler_name == "sqrt":
        scheduler = Schedulers(optimizer, lr, scheduler_name)
    elif scheduler_name == "inverse_time_decay":
        scheduler = Schedulers(optimizer, lr, scheduler_name)
    elif scheduler_name == "cosine_decay":
        scheduler = Schedulers(optimizer, lr, scheduler_name)
    elif scheduler_name == "exponential_decay":
        scheduler = Schedulers(optimizer, lr, scheduler_name)
    elif scheduler_name == 'step':
        scheduler = StepLR(optimizer, step_num)
    elif scheduler_name == "linear":
        scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=epochs)
    # elif scheduler_name == "cosine":
    #     self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
    elif scheduler_name == "exponential":
        scheduler = ExponentialLR(optimizer, 0.9)
    # elif scheduler_name == "plateau":
    #     self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
    elif scheduler_name == "cyclic":
        scheduler = OneCycleLR(optimizer, max_lr=initial_lr, steps_per_epoch=train_loader_len, epochs=epochs)
    else:
        raise ValueError("this scheduler is not supported")
    return scheduler
    