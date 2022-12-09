# SAM optimizer is impletented to the following algorithms

## 1. Backward Pass

For each d3rlpy/algos/torch/*_impl.py file, I found where `.step()` is used and added some code lines.  
E.g.) `dqn_impl.py`:

```python
######## For SAM ##########
if 'SAM' in self._optim_factory._optim_cls.__name__:
    def closure():
        self._optim.zero_grad()
        q_tpn = self.compute_target(batch)
        loss = self.compute_loss(batch, q_tpn)
        loss.backward()
        return loss
else:
    closure = None
###########################
...
loss.backward()
#self._optim.step()
######## For SAM ##########
self._optim.step(closure)
###########################
```

Most of the implementations uses only 1 `.step()`. Some of them have more than one.

- `awac_impl.py`
- `bc_impl.py`
- `bcq_impl.py`
- `bear_impl.py`: 3 `.step()`
- `cql_impl.py`
- `ddpg_impl.py`: 2 `.step()`
- `dqn_impl.py`: I tested on this file first. So far, `fit()` with SAM works without any error.
- `iql_impl.py`
- `plas_impl.py`
- `sac_impl.py`: 4 `.step()`

***

However, I couldn't find any `.step()` for some file(s):

- `combo_impl.py` -- based on CQL.
- `crr_impl.py` -- based on DDPG.
- `td3_impl.py` -- based on DDPG.
- `td3_plus_bc_impl.py` -- based on TD3.

Note: `fit()` function is inside of `d3rlpy/base.py`.

## 2. Loss Shaprness of minibatch (Foret et al. '21)

Sharpness as $\max_{\epsilon\in\mathcal{C}} L(w+\epsilon) - L(w)$.

1) For each d3rlpy/algos/torch/*_impl.py file, I found where `.step()` is used and added some code lines.  
E.g.) `bc_impl.py`:

```python
######## For SAM ##########
loss_sam = self._optim.step(closure)
if loss_sam is not None:
    loss_sharpness = loss_sam - loss.cpu().detach().numpy()
    return loss.cpu().detach().numpy(), loss_sharpness  # sharpness added!
###########################
```

This became possible due to the modification of SAM (`sam.py`):

```python
@torch.no_grad()
def step(self, closure=None):
    assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
    closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

    self.first_step(zero_grad=True)
    loss = closure()
    self.second_step() 
    return loss.cpu().detach().numpy()  # return L(w+eps).
```

2) For each d3rlpy/algos/*.py file, I found where `_update()` is defined and added some code lines.  
E.g.) `bc.py`:

```python
def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
    assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR
    loss = self._impl.update_imitator(batch.observations, batch.actions)
    ######## For SAM ##########
    if isinstance(loss, tuple):
        loss, sharpness = loss
        return {"loss":loss, "sharpness":sharpness}
    ###########################
    return {"loss": loss}
```
