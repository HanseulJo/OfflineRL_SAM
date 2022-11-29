# SAM optimizer is impletented to the following algorithms

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
