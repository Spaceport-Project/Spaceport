from statistics import mean
import os
import sys

import optuna
import torch
import numpy as np

from train import training, get_params
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from argparse import ArgumentParser, Namespace

# def objective(trial):
#     x = trial.suggest_float('x', -10, 10)
#     return (x - 2) ** 2

# study = optuna.create_study()
# study.optimize(objective, n_trials=100)

def objective(trial):
    # for optuna
    pass

def args_modification(args, model_params):
    args.expname = "optuna_test"
    args.render = True
    args.iterations = 2000 # fine
    args.coarse_iterations = 10000 # coarse
    
    # FOR OPACITY 
    # opacity_threshold_coarse, float. default -> 0.005
    # opacity_threshold_fine_init, float. default -> 0.005
    # opacity_reset_interval, int. default -> 3000
    # opacity_threshold_fine_after float. default -> 0.005

    return args

if __name__ == "__main__":
    script_name = sys.argv[0]
    print(f"Executed By -> {script_name}")
    args, model_params, opt_params, pipeline_params, model_hidden_params = get_params()
    args = args_modification(args, model_params)
    
    psnr_test_list_coarse, psnr_test_list_fine = training(model_params.extract(args), model_hidden_params.extract(args), opt_params.extract(args), pipeline_params.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)
    
    print("PSNR Coarse Output List Mean -> ", mean(psnr_test_list_coarse))
    print("PSNR Fine Output List Mean -> ", mean(psnr_test_list_fine))

    if args.render:
        try:
            print("\nRender after training..")
            from render import safe_state, render_sets
            safe_state(args.quiet)
            render_sets(model_params.extract(args), model_hidden_params.extract(args), -1, pipeline_params.extract(args), skip_train=True, skip_test=False, skip_video=False)
            print("\Rendering complete.")
        except Exception as e:
            print("Render Does not work due to -> ", e)

    # All done
    print("\nTraining complete.")