from argparse import ArgumentParser
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
import torch
from torch.optim import SGD, Adam
from tqdm import tqdm

import d3rlpy
from d3rlpy.datasets import get_dataset
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.metrics import evaluate_on_environment, td_error_scorer, discounted_sum_of_advantage_scorer, average_value_estimation_scorer
from sam import SAM

ALG_OPT_PARAM = {
    'AWAC': ['actor_optim_factory', 'critic_optim_factory'],
    'BCQ': ['actor_optim_factory', 'critic_optim_factory', 'imitator_optim_factory'],
    'BEAR': ['actor_optim_factory', 'critic_optim_factory', 'imitator_optim_factory'],
    'CQL': ['actor_optim_factory', 'critic_optim_factory'],
    'DDPG': ['actor_optim_factory', 'critic_optim_factory'],
    'IQL': ['actor_optim_factory', 'critic_optim_factory'],
    'SAC': ['actor_optim_factory', 'critic_optim_factory'],
    'DiscreteSAC':  ['actor_optim_factory', 'critic_optim_factory'],
    'TD3': ['actor_optim_factory', 'critic_optim_factory'],
    'TD3PlusBC': ['actor_optim_factory', 'critic_optim_factory'],
}

ALG_LR_KWARGS = {
    'AWAC': {'actor_learning_rate':3e-4, 'critic_learning_rate':3e-4},
    'BC': {'learning_rate':1e-3},
    'DiscreteBC': {'learning_rate':1e-3},
    'BCQ': {'actor_learning_rate':1e-3, 'critic_learning_rate':1e-3, 'imitator_learning_rate':1e-3},
    'DiscreteBCQ': {'learning_rate':6.25e-5},
    'BEAR': {'actor_learning_rate':1e-4, 'critic_learning_rate':3e-4, 'imitator_learning_rate':3e-4},
    'CQL': {'actor_learning_rate':1e-4, 'critic_learning_rate':3e-4},
    'DiscreteCQL': {'learning_rate':6.25e-5},
    'DDPG': {'actor_learning_rate':3e-4,  'critic_learning_rate':3e-4},
    'DQN': {'learning_rate':6.25e-5},
    'DoubleDQN': {'learning_rate':6.25e-5},
    'IQL': {'actor_learning_rate':3e-4, 'critic_learning_rate':3e-4},
    'SAC': {'actor_learning_rate':3e-4, 'critic_learning_rate':3e-4},
    'DiscreteSAC': {'actor_learning_rate':3e-4, 'critic_learning_rate':3e-4},
    'TD3': {'actor_learning_rate':3e-4, 'critic_learning_rate':3e-4},
    'TD3PlusBC': {'actor_learning_rate':3e-4, 'critic_learning_rate':3e-4},
}

def fix_seeds(random_seed):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == '__main__':
    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument('task_name',
        help = "Example: cartpole-replay, cartpole-random, pendulum-replay, pendulum-random, hopper-medium-v0, ..."
    )
    parser.add_argument('algorithm_name',
        help = "For discrete tasks, use: DiscreteBC, DiscreteBCQ, DiscreteCQL, DQN, DoubleDQN, DiscreteSAC."\
               " | For continuous tasks, use: AWAC, BC, BCQ, BEAR, CQL, DDPG, IQL, SAC, TD3, TD3PlusBC."
    )
    parser.add_argument('optimizers', nargs='+',
        help = "For BC & Discrete*, 1 argument (optim_name)."\
               " | For BCQ and BEAR, 3 arguments (actor_optim_name, critic_optim_name, imitator_optim_name)."\
               " | Otherwise, 2 argument (actor_optim_name, critic_optim_name)."\
               "Available: SGD, MomentumSGD, Adam, SamSGD, ASamSGD, SamMomentumSGD, ASamMomentumSGD, SamAdam, ASamSGD."
    )
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-r', '--rho', default=None, type=float, help="hparam for SAM or ASAM optimizers.")
    parser.add_argument('-e', '--n_epochs', default=None, type=int, help="Caveat: Use only either --n_epochs or --n_steps, but not both.")
    parser.add_argument('-s', '--n_steps', default=None, type=int, help="Caveat: Use only either --n_epochs or --n_steps, but not both.")
    parser.add_argument('-l', '--logging_num', default=10, type=int, help="How many times do you want to evaluate metrics?")
    parser.add_argument('-i', '--save_interval', default=1, type=int, help="How many times do you want to store the models?")
    parser.add_argument('-v', '--n_eval', default=100, type=int, help="How many times do you want to run the deployment?")
    parser.add_argument('-g', '--gpu', default=None, type=int)
    parser.add_argument('-p', '--pretrained_path', default=None)
    parser.add_argument('-T', '--tags', type=str, help="Add tag to d3rlpy_logs/*/.")
    parser.add_argument('-S', '--seed', default=None, type=int, help="Random seed for Train-Validation-split and Model training.")
    parser.add_argument('--lr_scale', default=None, type=float, help="Multiply some number to the default learning rates.")
    parser.add_argument('--lr_scale_SGD', default=None, type=float, help="Multiply some number to the default learning rates, only for SGD.")
    parser.add_argument('--verbose', action='store_true', help="Show the evaluated metrics every time. (number of printing: --logging_num)")
    parser.add_argument('--show_progress', action='store_true', help="Show the progress bar (tqdm). (number of printing: --logging_num)")
    parser.add_argument('-H', '--hessian_ckpt', nargs='+', default=None, type=int,
        help="What epoch do you want to calculate the hessian spectra (with SLQ)? Example: `--hessian_ckpt 0 1 -1` (before training, first epoch of training, and the last epoch)")
    parser.add_argument('--hessian_eval_num', default=None, type=int,
        help="How many times do you want to run Lanczos Algorithm in SLQ? (usually 100)"
    )
    parser.add_argument('--hessian_max_eig_every_epoch', action='store_true',
        help="If you want to evaluate hess. max. eig at every epoch, please turn on this option."
    )
    args = parser.parse_args()
    
    #use_gpu = torch.cuda.is_available()

    # Load dataset & environment
    dataset, env = get_dataset(args.task_name)
    env.reset()

    # train / validation split
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2, random_state=args.seed)

    # get algorithm type
    algorithm = getattr(d3rlpy.algos, args.algorithm_name)

    # Fix seeds
    if args.seed is not None:
        d3rlpy.seed(args.seed)
    
    # Default optimizers 
    OPT = {
        'SGD': OptimizerFactory(optim_cls=SGD, weight_decay=1e-4),
        'MomentumSGD': OptimizerFactory(optim_cls=SGD, momentum=0.9, weight_decay=1e-4),
        'Adam': OptimizerFactory(optim_cls=Adam, weight_decay=1e-4),
        'SamSGD': OptimizerFactory(optim_cls=SAM, base_optimizer="SGD", rho=0.001 if args.rho is None else args.rho, weight_decay=1e-4), 
        'SamMomentumSGD': OptimizerFactory(optim_cls=SAM, base_optimizer="SGD", momentum=0.9, rho=0.001 if args.rho is None else args.rho, weight_decay=1e-4), 
        'SamAdam': OptimizerFactory(optim_cls=SAM, base_optimizer="Adam", rho=0.001 if args.rho is None else args.rho, weight_decay=1e-4), 
        'ASamSGD': OptimizerFactory(optim_cls=SAM, base_optimizer="SGD", adaptive=True, rho=0.01 if args.rho is None else args.rho, weight_decay=1e-4), 
        'ASamMomentumSGD': OptimizerFactory(optim_cls=SAM, base_optimizer="SGD", momentum=0.9, adaptive=True, rho=0.01 if args.rho is None else args.rho, weight_decay=1e-4), 
        'ASamAdam': OptimizerFactory(optim_cls=SAM, base_optimizer="Adam", adaptive=True, rho=0.01 if args.rho is None else args.rho, weight_decay=1e-4),
    }
    MISC_OPT_KWARGS = {
        'temp_optim_factory': OPT['Adam'],
        'alpha_optim_factory': OPT['Adam']
    }

    # optimizers
    opt_zip = list(zip(ALG_OPT_PARAM.get(args.algorithm_name, ['optim_factory']), args.optimizers))
    opt_kwargs = {opt_type: OPT[opt_name] for opt_type, opt_name in opt_zip}
    opt_kwargs.update(MISC_OPT_KWARGS)
    opt_string = ''
    for opt_type, opt_name in opt_zip:
        opt_string += '_' + opt_type.split('_')[0] + '_' + opt_name
    if args.rho is not None:
        opt_string += f'_Rho{args.rho}'
    if args.lr_scale is not None:
        opt_string += f'_LRx{args.lr_scale}'
    elif args.lr_scale_SGD is not None:
        opt_string += f'_LRx{args.lr_scale_SGD}'
    opt_string = opt_string[1:]
    #opt_string = '_'.join(sum([[opt_type.split('_')[0], opt_name] for opt_type, opt_name in opt_zip], start=[]))  # only works for higher Python version

    # learning rates
    lr_kwargs = ALG_LR_KWARGS[args.algorithm_name]
    if args.lr_scale is not None:
        for k_opt, k_lr in zip(args.optimizers, lr_kwargs.keys()):
            lr_kwargs[k_lr] = lr_kwargs[k_lr] * args.lr_scale  # scale all LR
    elif args.lr_scale_SGD is not None:
        for k_opt, k_lr in zip(args.optimizers, lr_kwargs.keys()):
            if k_opt in ['SGD', 'SamSGD', 'ASamSGD']:
                lr_kwargs[k_lr] = lr_kwargs[k_lr] * args.lr_scale_SGD  # larger LR for SGD

    # build model
    model = algorithm(
        use_gpu=args.gpu if args.gpu is not None else torch.cuda.is_available(),
        batch_size=args.batch_size, **lr_kwargs, **opt_kwargs)
    if args.pretrained_path:
        model.build_with_env(env)
        model.load_model(args.pretrained_path)

    # logging directory & experiment name
    experiment_name = f"{args.algorithm_name}_{opt_string}"
    logdir = os.path.join("d3rlpy_logs", args.task_name+'_'+args.tags)

    # scorers
    scorers = {'environment': evaluate_on_environment(env)}
    if args.algorithm_name not in ['BC', 'DiscreteBC']:
        scorers.update({
            'td_error': td_error_scorer, # smaller is better
            'advantage': discounted_sum_of_advantage_scorer, # smaller is better
            'value_scale': average_value_estimation_scorer # smaller is better
        })

    # Train
    model.fit(
        train_episodes,
        eval_episodes=test_episodes,
        logdir=logdir,
        experiment_name=experiment_name,
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        n_steps_per_epoch=None if (args.n_steps is None or args.logging_num is None) else args.n_steps//args.logging_num,
        scorers=scorers,
        tensorboard_dir=logdir,
        verbose=args.verbose,
        show_progress=args.show_progress,
        save_interval=args.save_interval,
        n_eval=args.n_eval,
        hessian_ckpt=args.hessian_ckpt,
        hessian_eval_num=args.hessian_eval_num,
        hessian_max_eig_every_epoch=args.hessian_max_eig_every_epoch,
    )

    #if args.n_eval > 0 :
    #    print('\n Deployment:')
    #    scores=[evaluate_on_environment(env)(model) for _ in tqdm(range(args.n_eval))]
    #    print(f"Deployment score: Mean {np.mean(scores):.6f} std {np.std(scores):.6f}\n\n")
