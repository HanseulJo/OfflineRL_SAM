from argparse import ArgumentParser
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.optim import SGD, Adam
from tqdm import tqdm

import d3rlpy
from d3rlpy.datasets import get_dataset
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.metrics import evaluate_on_environment, td_error_scorer, discounted_sum_of_advantage_scorer, average_value_estimation_scorer
from sam import SAM


ALG_OPT_PARAM = {
    'DDPG': ['actor_optim_factory', 'critic_optim_factory'],
    'SAC': ['actor_optim_factory', 'critic_optim_factory'],
    'TD3': ['actor_optim_factory', 'critic_optim_factory'],
    'BCQ': ['actor_optim_factory', 'critic_optim_factory', 'imitator_optim_factory'],
    'BEAR': ['actor_optim_factory', 'critic_optim_factory', 'imitator_optim_factory'],
    'CQL': ['actor_optim_factory', 'critic_optim_factory'],
    'AWAC': ['actor_optim_factory', 'critic_optim_factory'],
    'IQL': ['actor_optim_factory', 'critic_optim_factory'],
}


if __name__ == '__main__':
    # Argument parsing
    parser = ArgumentParser()
    parser.add_argument('task_name')
    parser.add_argument('algorithm_name')
    parser.add_argument('optimizers', nargs='+',
        help="for BC & Discrete*, 1 argument (optim_name).\n"\
             "for BCQ and BEAR, 3 arguments (actor_optim_name, critic_optim_name, imitator_optim_name).\n"\
             "otherwise, 2 argument (actor_optim_name, critic_optim_name)."
    )
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-r', '--rho', default=None, type=float)
    parser.add_argument('-e', '--n_epochs', default=None, type=int)
    parser.add_argument('-s', '--n_steps', default=None, type=int)
    parser.add_argument('-l', '--logging_num', default=100, type=int)
    parser.add_argument('-v', '--n_eval', default=500, type=int)
    parser.add_argument('-g', '--use_gpu', default=False, type=int)
    parser.add_argument('-p', '--pretrained_path', default=None)
    parser.add_argument('-T', '--tags')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--show_progress', action='store_true')
    args = parser.parse_args()
    
    # Load dataset & environment
    dataset, env = get_dataset(args.task_name)
    env.reset()

    # train / validation split
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2, random_state=0)

    # get algorithm type
    algorithm = getattr(d3rlpy.algos, args.algorithm_name)
    
    
    # Default optimizers 
    OPT = {
        'SGD': OptimizerFactory(optim_cls=SGD),
        'Adam': OptimizerFactory(optim_cls=Adam),
        'SamSGD': OptimizerFactory(optim_cls=SAM, base_optimizer="SGD", rho=0.05 if args.rho is None else args.rho), # Foret et al. (2020)
        'SamAdam': OptimizerFactory(optim_cls=SAM, base_optimizer="Adam", rho=0.05 if args.rho is None else args.rho), # Foret et al. (2020)
        'ASamSGD': OptimizerFactory(optim_cls=SAM, base_optimizer="SGD", adaptive=True, rho=1.0 if args.rho is None else args.rho), # Kwon et al. (2021)
        'ASamAdam': OptimizerFactory(optim_cls=SAM, base_optimizer="Adam", adaptive=True, rho=1.0 if args.rho is None else args.rho), # Kwon et al. (2021)
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
    opt_string = opt_string[1:]
    #opt_string = '_'.join(sum([[opt_type.split('_')[0], opt_name] for opt_type, opt_name in opt_zip], start=[]))  # only works for higher Python version

    experiment_name = f"{args.algorithm_name}_{opt_string}"
    model = algorithm(use_gpu=args.use_gpu, **opt_kwargs)
    
    if args.pretrained_path:
        model.build_with_env(env)
        model.load_model(args.pretrained_path)

    # Train
    scorers = {'environment': evaluate_on_environment(env)}
    if args.algorithm_name not in ['BC', 'DiscreteBC']:
        scorers.update({
            'td_error': td_error_scorer, # smaller is better
            'advantage': discounted_sum_of_advantage_scorer, # smaller is better
            'value_scale': average_value_estimation_scorer # smaller is better
        })

    task_name_ = args.task_name+'_'+args.tags
    model.fit(
        train_episodes,
        eval_episodes=test_episodes,
        logdir=os.path.join("d3rlpy_logs",task_name_),
        experiment_name=experiment_name,
        n_epochs=args.n_epochs,
        n_steps=args.n_steps,
        n_steps_per_epoch=None if (args.n_steps is None or args.logging_num is None) else args.n_steps//args.logging_num,
        scorers=scorers,
        tensorboard_dir=os.path.join("tensorboard", task_name_),
        verbose=args.verbose,
        show_progress=args.show_progress,
        save_interval=1,
    )

    if args.n_eval > 0 :
        print('\n Deployment:')
        scores=[evaluate_on_environment(env)(model) for _ in tqdm(range(args.n_eval))]
        print(f"Deployment score: Mean {np.mean(scores):.6f} std {np.std(scores):6f}")

