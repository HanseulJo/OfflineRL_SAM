# This file manually inserted directly to the d3rlpy (Hanseul Cho, 2022.11.24)
#
## original source is https://github.com/davda54/sam/blob/main/sam.py
#
# Usage 1: 
# loss = loss_fn(image_preds, image_labels) # first forward pass
# loss.backward() # first backward pass
# optimizer.first_step(zero_grad=True)
# loss_fn(model(imgs), image_labels).backward() # second forward-backward pass
# optimizer.second_step(zero_grad=True)
#
# Usage 2:
# def closure():
#     loss = loss_function(output, model(input))
#     loss.backward()
#     return loss
# loss = loss_function(output, model(input))
# loss.backward()
# optimizer.step(closure)
# optimizer.zero_grad()


import torch
from torch import optim
from typing import Type, cast, Union

from torch import optim
from torch.optim import *


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer: Union[Optimizer, str], rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        defaults = dict(rho=rho, adaptive=self.adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        if isinstance(base_optimizer, str):
            self.base_optimizer = cast(Type[Optimizer], getattr(optim, base_optimizer))(self.param_groups, **kwargs)
        else:
            self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step() 
        return loss  # return L_B(w+eps).

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups