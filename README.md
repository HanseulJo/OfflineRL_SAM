Source code for AI611 course project: "Can Sharpness-Aware Training be Beneficial for Offline RL?"

Final report in Notion blog post: https://valiant-tadpole-f7f.notion.site/Can-Sharpness-Aware-Training-be-Beneficial-for-Offline-RL-d42e4f8195c54f07a2c15673bd7b6c68

The main contribution of this repository:
- To make SAM optimizer available for every RL algorithms embedded in [d3rlpy](https://github.com/takuseno/d3rlpy)
- To make the Hessian maximum eigenvalue (HME) and empirical spectral density of Hessian (ESD) available during training on offline reiniforcement learning dataset. (Based on [PyHessian](https://github.com/amirgholami/PyHessian))

## `Experiment.py` running example

```bash
$ python experiment.py -h
usage: experiment.py [-h] [-b BATCH_SIZE] [-r RHO] [-e N_EPOCHS] [-s N_STEPS] [-l LOGGING_NUM] [-i SAVE_INTERVAL] [-v N_EVAL] [-p PRETRAINED_PATH] [-T TAGS] [-S SEED] [--lr_scale LR_SCALE] [--lr_scale_SGD LR_SCALE_SGD] [--verbose] [--show_progress] [-H HESSIAN_CKPT [HESSIAN_CKPT ...]] [--hessian_eval_num HESSIAN_EVAL_NUM] task_name algorithm_name optimizers [optimizers ...]

positional arguments:
  task_name             Example: cartpole-replay, cartpole-random, pendulum-replay, pendulum-random, hopper-medium-v0, ...
  algorithm_name        For discrete tasks, use: DiscreteBC, DiscreteBCQ, DiscreteCQL, DQN, DoubleDQN, DiscreteSAC.
                        For continuous tasks, use: AWAC, BC, BCQ, BEAR, CQL, DDPG, IQL, SAC, TD3, TD3PlusBC.
  optimizers            For BC & Discrete*, 1 argument (optim_name). 
                        For BCQ and BEAR, 3 arguments (actor_optim_name, critic_optim_name, imitator_optim_name). 
                        Otherwise, 2 argument (actor_optim_name, critic_optim_name).Available: SGD, MomentumSGD, Adam, SamSGD, ASamSGD,SamMomentumSGD, ASamMomentumSGD, SamAdam, ASamSGD.

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -r RHO, --rho RHO     hparam for SAM or ASAM optimizers.
  -e N_EPOCHS, --n_epochs N_EPOCHS
                        Caveat: Use only either --n_epochs or --n_steps, but not both.
  -s N_STEPS, --n_steps N_STEPS
                        Caveat: Use only either --n_epochs or --n_steps, but not both.
  -l LOGGING_NUM, --logging_num LOGGING_NUM
                        How many times do you want to evaluate metrics?
  -i SAVE_INTERVAL, --save_interval SAVE_INTERVAL
                        How many times do you want to store the models?
  -v N_EVAL, --n_eval N_EVAL
                        How many times do you want to run the deployment?
  -p PRETRAINED_PATH, --pretrained_path PRETRAINED_PATH
  -T TAGS, --tags TAGS  Add tag to d3rlpy_logs/*/.
  -S SEED, --seed SEED  Random seed for train-validation split.
  --lr_scale LR_SCALE   Multiply some number to the default learning rates.
  --lr_scale_SGD LR_SCALE_SGD
                        Multiply some number to the default learning rates, only for SGD.
  --verbose             Show the evaluated metrics every time. (number of printing: --logging_num)
  --show_progress       Show the progress bar (tqdm). (number of printing: --logging_num)
  -H HESSIAN_CKPT [HESSIAN_CKPT ...], --hessian_ckpt HESSIAN_CKPT [HESSIAN_CKPT ...]
                        What epoch do you want to calculate the hessian spectra (with SLQ)? Example: `--hessian ckpt 0 1 -1` (before training, first epoch of training, and the last epoch)
  --hessian_eval_num HESSIAN_EVAL_NUM
                        How many times do you want to run Lanczos Algorithm in SLQ?
 ```

Example:

```bash
python experiment.py pendulum-replay BC SGD --tag customTag --batch_size 100 --n_steps 100000 --logging_num 100 --save_interval 10 --show_progress --hessian_ckpt 0 1 -1 --hessian_eval_num 100 --lr_scale_SGD 100
```

```bash
python experiment.py pendulum-replay BC Adam --tag customTag --batch_size 100 --n_steps 100000  --show_progress --logging_num 100 --save_interval 10 --hessian_ckpt 0 1 -1 --hessian_eval_num 100
```

```bash
python experiment.py pendulum-replay BC SamSGD --rho 0.001 --tag customTag --batch_size 100 --n_steps 100000 --logging_num 100 --save_interval 10 --show_progress --hessian_ckpt 0 1 -1 --hessian_eval_num 100 --lr_scale_SGD 100
```

```bash
python experiment.py pendulum-replay BC ASamSGD --rho 0.001 --tag customTag --batch_size 100 --n_steps 100000 --logging_num 100 --save_interval 10 --show_progress --hessian_ckpt 0 1 -1 --hessian_eval_num 100 --lr_scale_SGD 100
```

```bash
python experiment.py pendulum-replay BC SamAdam --rho 0.001 --tag customTag --batch_size 100 --n_steps 100000 --logging_num 100 --save_interval 10 --show_progress --hessian_ckpt 0 1 -1 --hessian_eval_num 100
```

```bash
python experiment.py pendulum-replay BC ASamAdam --rho 0.001 --tag customTag --batch_size 100 --n_steps 100000 --logging_num 100 --save_interval 10 --show_progress --hessian_ckpt 0 1 -1 --hessian_eval_num 100
```
