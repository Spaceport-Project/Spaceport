from statistics import mean
import os
import sys
import time
import json 
import shutil
import gc

import optuna
import torch
import numpy as np

from train import training, get_params

# FOR OPACITY 
# opacity_threshold_coarse, float. default -> 0.005
# opacity_threshold_fine_init, float. default -> 0.005
# opacity_reset_interval, int. default -> 3000
# opacity_threshold_fine_after float. default -> 0.005

# FOR DENSITY
# densify_until_iter, int. default 15000
# densify_from_iter, int. default 500
# densification_interval, int. default 100
# densify_grad_threshold_after, float. default 0.0002
# densify_grad_threshold_fine_init, float default 0.0002
# densify_grad_threshold_coarse, float default 0.0002

# FOR DEFORMATION
# no_ds, bool. default False
# no_dr, bool. default false
# no_dx, bool. default False
# no_dshs, bool. default True
# net_width, int default 64
# defor_depth, int. default 1
# no_do, True. default True
# deformation_lr_init, float. default 0.00016
# deformation_lr_delay_mult, float. default 0.01
# deformation_lr_final, float. default 0.000016

# FOR MODEL PARAMS
# net_width int, default 64
# defor_depth int, default 1
# plane_tv_weight float, default 0.0001
# time_smoothness_weight float, default 0.01
# l1_time_planes float, default 0.0001

# FOR LOSS (WIP)
# L1, default loss function
# L2
# Lx + SSIM
# Lx + LPIPS

def all_params_obj(trial):
    torch.cuda.empty_cache()
    try:
        shutil.rmtree(f"/home/alper/Spaceport/output/Optimization_folder") # Do not change this name
        print("Removed Optimization_folder")
    except Exception as e:
        print("Cannot remove optmization folder -> ", e)
    
    args.expname = f"Optimization_folder" # Do not change this name

    args.deformation_lr_final = trial.suggest_float('deformation_lr_final', 0.000016, 0.0016)
    args.deformation_lr_init = trial.suggest_float('deformation_lr_init', 0.000016, 0.0016)
    args.deformation_lr_delay_mult = trial.suggest_float('deformation_lr_delay_mult', 0.001, 0.1)

    # args.no_do = trial.suggest_categorical("no_do", [True, False])
    # args.no_ds = trial.suggest_categorical("no_ds", [True, False])
    # args.no_dx = trial.suggest_categorical("no_dx", [True, False])
    # args.no_dshs = trial.suggest_categorical("no_dshs", [True, False])

    args.net_width = trial.suggest_categorical("net_width", [64, 128, 256, 512])
    args.defor_depth = trial.suggest_categorical("defor_depth", [1, 2, 4, 8, 16, 64])
    args.plane_tv_weight = trial.suggest_float('plane_tv_weight', 0.00001, 0.001)
    args.time_smoothness_weight = trial.suggest_float('time_smoothness_weight', 0.001, 0.1)
    args.l1_time_planes = trial.suggest_float('l1_time_planes', 0.00001, 0.001)

    args.densify_until_iter = trial.suggest_int('densify_until_iter', 10000, 20000)
    args.densify_from_iter = trial.suggest_int('densify_from_iter', 500, 2000)
    args.densification_interval = trial.suggest_int('densification_interval', 50, 500)
    args.densify_grad_threshold_after = trial.suggest_float('densify_grad_threshold_after', 0.00001, 0.01)
    args.densify_grad_threshold_fine_init = trial.suggest_float('densify_grad_threshold_fine_init', 0.00001, 0.01)
    args.densify_grad_threshold_coarse = trial.suggest_float('densify_grad_threshold_coarse', 0.00001, 0.01)

    args.opacity_threshold_coarse = trial.suggest_float('opacity_threshold_coarse', 0.0005, 0.1)
    args.opacity_threshold_fine_init = trial.suggest_float('opacity_threshold_fine_init', 0.0005, 0.1)
    args.opacity_threshold_fine_after = trial.suggest_float('opacity_threshold_fine_after', 0.0005, 0.1)
    args.opacity_reset_interval = trial.suggest_float('opacity_reset_interval', 500, 5000)
    
    psnr_test_list_coarse, psnr_test_list_fine = training(model_params.extract(args), model_hidden_params.extract(args), opt_params.extract(args), pipeline_params.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)
    gc.collect()
    torch.cuda.empty_cache()

    return mean(psnr_test_list_fine[-5:]) # mean of the last five items (iteration results)

def deformation_obj(trial):
    # deformation optimization
    args.deformation_lr_final = trial.suggest_float('deformation_lr_final', 0.000016, 0.0016)
    args.deformation_lr_init = trial.suggest_float('deformation_lr_init', 0.000016, 0.0016)
    args.deformation_lr_delay_mult = trial.suggest_float('deformation_lr_delay_mult', 0.001, 0.1)

    args.no_do = trial.suggest_categorical("no_do", [True, False])
    args.no_ds = trial.suggest_categorical("no_ds", [True, False])
    args.no_dx = trial.suggest_categorical("no_dx", [True, False])
    args.no_dshs = trial.suggest_categorical("no_dshs", [True, False])

    args.net_width = trial.suggest_categorical("net_width", [64, 128, 256, 512])
    args.defor_depth = trial.suggest_categorical("defor_depth", [1, 2, 4, 8, 16, 64])
    
def density_obj(trial):
    # density optimization
    args.densify_until_iter = trial.suggest_int('densify_until_iter', 10000, 20000)
    args.densify_from_iter = trial.suggest_int('densify_from_iter', 500, 2000)
    args.densification_interval = trial.suggest_int('densification_interval', 50, 500)

    args.densify_grad_threshold_after = trial.suggest_float('densify_grad_threshold_after', 0.00001, 0.01)
    args.densify_grad_threshold_fine_init = trial.suggest_float('densify_grad_threshold_fine_init', 0.00001, 0.01)
    args.densify_grad_threshold_coarse = trial.suggest_float('densify_grad_threshold_coarse', 0.00001, 0.01)

def opacity_obj(trial):
    # for optuna
    try:
        shutil.rmtree(f"/home/alper/Spaceport/output/Optimization_folder") # Do not change this name
        print("Removed Optimization_folder")
    except Exception as e:
        print("Cannot remove optmization folder -> ", e)
        
    args.opacity_threshold_coarse = trial.suggest_float('opacity_threshold_coarse', 0.0005, 0.1)
    args.opacity_threshold_fine_init = trial.suggest_float('opacity_threshold_fine_init', 0.0005, 0.1)
    args.opacity_threshold_fine_after = trial.suggest_float('opacity_threshold_fine_after', 0.0005, 0.1)
    args.opacity_reset_interval = trial.suggest_float('opacity_reset_interval', 500, 5000)
    
    args.expname = f"Optimization_folder" # Do not change this name

    psnr_test_list_coarse, psnr_test_list_fine = training(model_params.extract(args), model_hidden_params.extract(args), opt_params.extract(args), pipeline_params.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)
    return mean(psnr_test_list_fine[-5:]) # mean of the last five items (iteration results)

def vanilla_train(expname, iterations, coarse_iterations):
    script_name = sys.argv[0]
    print(f"Executed By -> {script_name}")
    args, model_params, opt_params, pipeline_params, model_hidden_params = get_params()
    args.expname = f"{expname}_vanilla"
    args.render = False
    args.iterations = iterations # fine
    args.coarse_iterations = coarse_iterations # coarse
    args.quiet = False
    psnr_test_list_coarse, psnr_test_list_fine = training(model_params.extract(args), model_hidden_params.extract(args), opt_params.extract(args), pipeline_params.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)

    avg_psnr = mean(psnr_test_list_fine[-5:])

    return avg_psnr

def render_function(args):
    print("Rendering Function Called..")
    try:
        print("\nRender after training..")
        from render import safe_state, render_sets
        safe_state(args.quiet)
        render_sets(model_params.extract(args), model_hidden_params.extract(args), -1, pipeline_params.extract(args), skip_train=True, skip_test=False, skip_video=False)
        print("\Rendering complete.")
    except Exception as e:
        print("Render Does not work due to -> ", e)

if __name__ == "__main__":
    script_name = sys.argv[0]
    args, model_params, opt_params, pipeline_params, model_hidden_params = get_params()
    
    trial_no = 50
    args.expname = "optuna_comparision"
    args.render = False
    args.iterations = 30000 # fine
    args.coarse_iterations = 7000 # coarse

    trial_start_time = time.time()
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///db.sqlite3",
                                study_name="opacity_optimization_db")
    
    study.optimize(lambda trial: all_params_obj(trial), n_trials=trial_no)

    trial_end_time = time.time()
    trial_elapsed_time = trial_end_time - trial_start_time
    trial_elapsed_time_minutes = trial_elapsed_time / 60
    
    start_time = time.time()

    vanilla_psnr = vanilla_train(expname=args.expname, iterations=args.iterations, coarse_iterations=args.coarse_iterations)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_minutes = elapsed_time / 60

    print(f"\nTrial Trains Time Consumption -> {trial_elapsed_time_minutes:.2f} minute.")
    print(f"Vanilla Trains Time Consumption -> {elapsed_time_minutes:.2f} minute.")

    print("\nVanilla PSNR Value -> ", vanilla_psnr)
    print("Best Trial PSNR -> ", study.best_value)
    print("--"*10)
    print("\nOptimized params ->", study.best_params)

    summary = study.best_params
    summary['trial_psnr'] = study.best_value
    summary['vanilla_psnr'] = vanilla_psnr

    output_json_path = os.path.join(os.path.join("./output/", args.expname), "summary.json") # /output/{expname}/summary.json

    try:
        with open(output_json_path, "w") as outfile: 
            json.dump(summary, outfile)
    except Exception as e:
        print(f"Json Has Not Written Properly Due to {e}")