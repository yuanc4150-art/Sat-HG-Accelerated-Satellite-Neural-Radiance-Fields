Sat-HG: Accelerated Satellite Neural Radiance Fields
This repository contains the official PyTorch implementation of Sat-HG.
It provides an accelerated and memory-efficient extension of Sat-NeRF using HashGrid encoding (Instant-NGP). By leveraging tiny-cuda-nn, robust UTM coordinate transformations, and memory-safe chunk rendering, Sat-HG achieves faster training, rendering, and high-quality 3D surface reconstruction from satellite imagery.
1. Setup and Installation
The code has been tested with Python 3.9 and PyTorch 2.4.0 (CUDA 11.8).
Step 1: Clone the repository
code
Bash
git clone https://github.com/YOUR_USERNAME/sat-hg.git
cd sat-hg
Step 2: Install dependencies (Recommended via Conda)
We highly recommend using Conda to set up the environment, as GDAL can be difficult to install via pip on some systems:
code
Bash
conda env create -f environment.yml
conda activate satngp
Step 3: Install tinycudann (Crucial for HashGrid)
Due to CUDA compilation requirements, tiny-cuda-nn must be installed manually. Please ensure your nvcc --version matches the PyTorch CUDA version, then run:
code
Bash
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
2. Dataset Preparation
The original images originate from the DFC2019 dataset. For convenience, we use the pre-processed version provided by the Sat-NeRF repository.
Run the following commands to download and extract the dataset:
code
Bash
mkdir data
cd data
wget https://github.com/centreborelli/satnerf/releases/download/EarthVision2022/dataset.zip
unzip dataset.zip -d dataset
cd ..
To simplify the execution commands for training and evaluation, please set the path to your dataset and desired output directory as environment variables:
code
Bash
export DATA_DIR=$(pwd)/data/dataset
export OUT_DIR=/path/to/your/sathg-output
3. Training
Use main.py to train the Sat-HG model. The following command provides an example of training on the JAX_260 AOI with full features enabled (appearance embedding, radiometric normalization, distortion loss, etc.).
code
Bash
python3 main.py \
    --model sat-nerf-ngp \
    --exp_name JAX_260_lr0.005 \
    --gpu_id 0 \
    --root_dir $DATA_DIR/root_dir/crops_rpcs_ba_v2/JAX_260 \
    --img_dir $DATA_DIR/DFC2019/Track3-RGB-crops/JAX_260 \
    --cache_dir $DATA_DIR/DFC2019/cache_dir/JAX_260_ngp \
    --gt_dir $DATA_DIR/DFC2019/Track3-Truth \
    --logs_dir $OUT_DIR/logs \
    --ckpts_dir $OUT_DIR/ckpts \
    --lr 0.005 \
    --batch_size 1024 \
    --chunk 4096 \
    --n_samples 64 \
    --n_importance 64 \
    --max_train_steps 100000 \
    --use_smoothness_loss \
    --use_appearance_embedding \
    --use_distortion_loss \
    --radiometric_normalization \
    --ds_lambda 0.1 \
    --sc_lambda 0.1 \
    --img_downscale 2.0
Note: A timestamp will be automatically prepended to your --exp_name (e.g., 2026-05-11_15-24-41_JAX_260_lr0.005). You will need this full identifier for the evaluation step.
4. Evaluation and DSM Generation
Use eval_sathg.py to evaluate the trained model, compute quantitative metrics (PSNR, SSIM), and extract the Digital Surface Model (DSM).
Please replace 2026-05-11_15-24-41_JAX_260_lr0.005 with your actual generated experiment ID, and 46 with your target epoch number.
code
Bash
python3 eval_sathg.py \
    2026-05-11_15-24-41_JAX_260_lr0.005 \
    $OUT_DIR/logs \
    $OUT_DIR/eval_results \
    46 \
    test \
    --checkpoints_dir $OUT_DIR/ckpts \
    --root_dir $DATA_DIR/root_dir/crops_rpcs_ba_v2/JAX_260 \
    --img_dir $DATA_DIR/DFC2019/Track3-RGB-crops/JAX_260 \
    --gt_dir $DATA_DIR/DFC2019/Track3-Truth
Expected Output: The script will load the specified checkpoint, render the images/depth maps in chunks without OOM, save the outputs to $OUT_DIR/eval_results, and print the final evaluation metrics.
Acknowledgements
This project is built upon the foundational architecture of the official Sat-NeRF repository. We sincerely thank the original authors for their excellent open-source work.
Our main structural modifications upon their codebase include:
Integration of tiny-cuda-nn for HashGrid representation.
Re-engineering of the ECEF-to-UTM projection pipeline using modern pyproj APIs.
Refactoring of the chunk-based volume rendering logic to eliminate Out-Of-Memory (OOM) issues during large-scale evaluation.
