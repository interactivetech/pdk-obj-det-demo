import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR


def WarmupWrapper(scheduler_type):
    class Wrapped(scheduler_type):
        def __init__(self, warmup, warmup_iters, warmup_ratio, *args):
            self.warmup = warmup
            if warmup is not None:
                assert warmup in ["constant", "linear", "exp"]
                assert warmup_iters > 0
                assert 0 < warmup_ratio <= 1.0
                self.warmup_iters = warmup_iters
                self.warmup_ratio = warmup_ratio
            super(Wrapped, self).__init__(*args)

        def get_warmup_mult(self):
            if self.warmup == "constant":
                mult = self.warmup_ratio
            elif self.warmup == "linear":
                mult = self.warmup_ratio + (self.last_epoch + 1) / self.warmup_iters * (
                    1 - self.warmup_ratio
                )
            elif self.warmup == "exp":
                mult = self.warmup_ratio ** (
                    1 - (self.last_epoch + 1) / self.warmup_iters
                )
            else:
                raise NotImplementedError
            return mult

        def get_lr(self):
            if self.last_epoch < self.warmup_iters:
                mult = self.get_warmup_mult()
                return [mult * b_lr for b_lr in self.base_lrs]
            return super(Wrapped, self).get_lr()

    return Wrapped


if __name__ == "__main__":
    w = torch.randn(1, requires_grad=True)
    loss = torch.nn.MSELoss()
    optimizer = SGD([w], 0.01, 0.0004)
    scheduler_cls = WarmupWrapper(MultiStepLR)
    scheduler = scheduler_cls("linear", 1000, 0.001, optimizer, [118272,162624], 0.1)
    warmup_plot = 1000
    for i in range(192192):
        x = torch.randn(100)
        if i <warmup_plot and i%100==0:
            print(i, " 1-lr: ",scheduler.get_lr())
        elif i>warmup_plot and i%10000==0:
            print(i, " 2-lr: ",scheduler.get_lr())
        y = 3 * x
        output = loss(w * x, y)
        output.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
