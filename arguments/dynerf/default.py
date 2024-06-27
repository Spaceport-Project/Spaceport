ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 25]
    },
    multires = [1,2,4,8],
    defor_depth = 8,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    empty_voxel=False,
    render_process=False,
    static_mlp=False,
)

OptimizationParams = dict(

    dataloader=True,

    iterations = 30_000,
    batch_size=1,
    coarse_iterations = 7000,

    densify_until_iter = 15_000,
    opacity_lr=0.0005,
    opacity_reset_interval = 3000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000,
    # lambda_lpips=0.5,
    # lambda_dssim=0.5,
)