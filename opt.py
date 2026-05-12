import argparse
import datetime
import json
import os


def get_opts():
    parser = argparse.ArgumentParser()

    # --- Group 1: Path Arguments ---
    path_group = parser.add_argument_group('Paths', 'Input and output paths')
    path_group.add_argument('--root_dir', type=str, required=True,
                            help='Root directory of the input dataset (containing JSONs, train.txt, etc.)')
    path_group.add_argument('--img_dir', type=str, default=None,
                            help='Directory where the images are located (if different than root_dir)')
    path_group.add_argument('--gt_dir', type=str, default=None,
                            help='Directory where the ground truth DSM is located (if available)')
    path_group.add_argument('--cache_dir', type=str, default=None,
                            help='Directory to cache precomputed rays')
    path_group.add_argument("--ckpts_dir", type=str, default="ckpts",
                            help="Directory to save trained model checkpoints")
    path_group.add_argument("--logs_dir", type=str, default="logs",
                            help="Directory to save TensorBoard logs and visualizations")
    path_group.add_argument("--ckpt_path", type=str, default=None,
                            help="Path to a pretrained checkpoint to resume training from")

    # --- Group 2: Experiment and Model Selection ---
    exp_group = parser.add_argument_group('Experiment', 'General experiment settings')
    exp_group.add_argument("--exp_name", type=str, default=None,
                           help="Experiment name. A timestamp will be prepended.")
    exp_group.add_argument('--data', type=str, default='sat', choices=['sat', 'blender'],
                           help='Type of dataset')

    exp_group.add_argument("--model", type=str, default="sat-nerf-ngp",
                           choices=['nerf', 's-nerf', 'sat-nerf', 'sat-nerf-ngp'],
                           help="Which NeRF variant to use")
    exp_group.add_argument("--gpu_id", type=int, required=True,
                           help="ID of the GPU to use for training")

    # --- Group 3: Training Hyperparameters ---
    train_group = parser.add_argument_group('Training', 'Core training hyperparameters')

    train_group.add_argument('--lr', type=float, default=1e-2,
                             help='Initial learning rate (e.g., 1e-2 for NGP, 5e-4 for original models)')
    train_group.add_argument('--batch_size', type=int, default=4096,
                             help='Number of rays per training iteration. Decrease if out of memory.')
    train_group.add_argument('--chunk', type=int, default=16384,
                             help='Chunk size for processing rays. Decrease if out of memory during validation.')
    train_group.add_argument('--max_train_steps', type=int, default=150000,
                             help='Total number of training iterations')
    train_group.add_argument('--save_every_n_epochs', type=int, default=1,
                             help="Save checkpoints every N epochs")
    train_group.add_argument('--img_downscale', type=float, default=1.0,
                             help='Factor to downscale input images')

    # --- Group 4: NeRF Model Architecture ---
    arch_group = parser.add_argument_group('Architecture', 'NeRF model architecture settings')

    arch_group.add_argument('--fc_units', type=int, default=128,
                            help='Number of hidden units in MLPs or feature dimension from geometry network')
    arch_group.add_argument('--fc_layers', type=int, default=8,
                            help='Number of layers in the original NeRF MLP (ignored by NGP model)')
    arch_group.add_argument('--n_samples', type=int, default=128,
                            help='Number of coarse samples per ray')
    arch_group.add_argument('--n_importance', type=int, default=64,
                            help='Number of fine samples per ray (set > 0 to enable fine model)')
    arch_group.add_argument('--noise_std', type=float, default=0.0,
                            help='Std dev of noise added to sigma for regularization')

    # --- Group 5: Sat-NeRF Specific Arguments ---
    sat_group = parser.add_argument_group('Sat-NeRF Specific', 'Hyperparameters for Sat-NeRF and variants')

    # Loss-related switches
    sat_group.add_argument('--radiometric_normalization', action='store_true', default=False,
                           help='Enable radiometric affine transformation')

    # Embedding parameters
    sat_group.add_argument('--transient_embedding_dim', type=int, default=16,
                           help='Dimension of the transient embedding vector (t)')
    sat_group.add_argument('--appearance_embedding_dim', type=int, default=32,
                           help='Dimension of the appearance embedding vector (a)')
    sat_group.add_argument('--embedding_vocab_size', type=int, default=30,
                           help='Vocabulary size for embeddings (>= number of train images)')

    # Loss parameters
    sat_group.add_argument('--sc_lambda', type=float, default=0.1,
                           help='Weight for the solar correction loss')
    sat_group.add_argument('--ds_lambda', type=float, default=0.05,
                           help='Weight for the depth supervision loss')
    sat_group.add_argument('--ds_drop', type=float, default=0.5,
                           help='Portion of training steps to apply depth supervision')
    sat_group.add_argument('--first_beta_epoch', type=int, default=1,
                           help='Epoch to start using the uncertainty loss')
    sat_group.add_argument('--beta_lambda', type=float, default=0.01,
                           help='Weight for the log-beta regularization loss')
    sat_group.add_argument('--beta_min', type=float, default=1e-3,
                           help='Minimum clamp value for beta')
    sat_group.add_argument('--distortion_lambda', type=float, default=1e-4,
                           help='Weight for the distortion loss')
    sat_group.add_argument('--distortion_warmup_steps', type=int, default=20000,
                           help='Steps to warmup distortion loss')

    # Loss-related switches
    sat_group.add_argument('--ds_noweights', action='store_true',
                           help='Disable reprojection error weighting for depth loss')
    sat_group.add_argument('--use_distortion_loss', action='store_true', default=True,
                           help="Enable the distortion loss.")
    sat_group.add_argument('--no_distortion_loss', action='store_false', dest='use_distortion_loss',
                           help="Disable the distortion loss.")
    sat_group.add_argument('--use_smoothness_loss', action='store_true', default=True,
                           help="Enable the edge-aware smoothness loss.")
    sat_group.add_argument('--no_smoothness_loss', action='store_false', dest='use_smoothness_loss',
                           help="Disable the smoothness loss.")
    sat_group.add_argument('--use_charbonnier_loss', action='store_true', default=True,
                           help="Use robust Charbonnier loss for depth.")
    sat_group.add_argument('--no_charbonnier_loss', action='store_false', dest='use_charbonnier_loss',
                           help="Use standard L1 loss for depth.")
    sat_group.add_argument('--use_appearance_embedding', action='store_true', default=True,
                           help="Enable the appearance embedding.")
    sat_group.add_argument('--no_appearance_embedding', action='store_false', dest='use_appearance_embedding',
                           help="Disable the appearance embedding.")
    # --- Group 6: HashGrid Specific Arguments (for sat-nerf-ngp) ---
    hash_group = parser.add_argument_group('HashGrid', 'Hyperparameters for the HashGrid encoder')

    hash_group.add_argument('--use_hashgrid', action='store_true', default=True,
                            help="Use HashGrid encoding (default for sat-nerf-ngp)")
    hash_group.add_argument('--no_hashgrid', action='store_false', dest='use_hashgrid',
                            help="Disable HashGrid and fallback to original MLP (for debugging)")
    hash_group.add_argument('--hash_n_levels', type=int, default=16,
                            help='Number of levels in the hash grid')
    hash_group.add_argument('--hash_features_per_level', type=int, default=2,
                            help='Number of features per hash grid level')
    hash_group.add_argument('--hash_log2_hashmap_size', type=int, default=17,
                            help='Log2 of the hash map size')
    hash_group.add_argument('--hash_base_resolution', type=int, default=16,
                            help='Base resolution of the hash grid')
    hash_group.add_argument('--hash_per_level_scale', type=float, default=1.8,
                            help='Scaling factor between hash grid levels')

    args = parser.parse_args()

    # --- Post-process and setup experiment directories ---
    if args.exp_name is None:
        # Generate a default experiment name if not provided
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name_simple = args.model.replace('-', '_')
        dataset_name = os.path.basename(os.path.normpath(args.root_dir))
        args.exp_name = f"{timestamp}_{model_name_simple}_{dataset_name}"

    # REMOVED: No longer automatically prepending a timestamp.
    # The caller (e.g., an automation script) is now fully responsible for the experiment name.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.exp_name = f"{timestamp}_{args.exp_name}"

    print(f"\nRunning experiment: {args.exp_name} - Using gpu {args.gpu_id}\n")

    exp_log_dir = os.path.join(args.logs_dir, args.exp_name)
    os.makedirs(exp_log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.ckpts_dir, args.exp_name), exist_ok=True)

    with open(os.path.join(exp_log_dir, "opts.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return args
