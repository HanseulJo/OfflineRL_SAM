# OfflineRL_SAM
AI611 final project

## `Experiment.py` running example

```bash
python experiment.py pendulum-random BC ASamAdam --tag my_tag --n_steps 10000 --logging_num 100 --n_eval 100 --verbose --show_progress  # one optimizer
python experiment.py hopper-medium-v0 CQL Adam ASamAdam --tag 123 --n_steps 100000 --logging_num 10 --n_eval 100 --verbose --show_progress --use_gpu  # two optimizers
python experiment.py hopper-medium-v0 BCQ Adam SamAdam ASamAdam --tag custom_tag --n_steps 100000 --logging_num 10 --n_eval 100 --verbose --show_progress --use_gpu  # three optimizers
```

```bash
$ python experiment.py -h
usage: experiment.py [-h] [-b BATCH_SIZE] [-e N_EPOCHS] [-s N_STEPS] [-l LOGGING_NUM] [-v N_EVAL] [-g USE_GPU] [-p PRETRAINED_PATH] [-T TAGS] [--verbose] [--show_progress] task_name algorithm_name optimizers [optimizers ...]

positional arguments:
  task_name
  algorithm_name
  optimizers            For BC & Discrete*, 1 argument (optim_name). For BCQ and BEAR, 3 arguments (actor_optim_name, critic_optim_name, imitator_optim_name). Otherwise, 2 argument (actor_optim_name, critic_optim_name).

options:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -e N_EPOCHS, --n_epochs N_EPOCHS
  -s N_STEPS, --n_steps N_STEPS
  -l LOGGING_NUM, --logging_num LOGGING_NUM
  -v N_EVAL, --n_eval N_EVAL
  -g USE_GPU, --use_gpu USE_GPU
  -p PRETRAINED_PATH, --pretrained_path PRETRAINED_PATH
  -T TAGS, --tags TAGS
  --verbose
  --show_progress
 ```
