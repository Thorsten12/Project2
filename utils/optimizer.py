import torch

class StochasticGradientDescent(torch.optim.Optimizer):
    def __init__(self, params, lr=0.003, momentum=0.9, weight_decay=0): # These values are straight from the paper
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    @torch.no_grad()    # we need no gradient
    def step(self):
        for group in self.param_groups:
            lr          = group["lr"]
            momentum    = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad   # from loss.backward() in train.py

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.clone(grad).detach()
                else:
                    state["momentum_buffer"].mul_(momentum).add_(grad)

                p.add_(state["momentum_buffer"], alpha=-lr)

                