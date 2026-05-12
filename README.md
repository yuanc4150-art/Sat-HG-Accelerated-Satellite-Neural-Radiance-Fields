---

# Sat-HG

This repository contains the official PyTorch implementation of **Sat-HG**. 

---

## 1. Setup and Installation

The code has been tested with **Python 3.9** and **PyTorch 2.4.0** .

**Step 1: Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/sat-hg.git
cd sat-hg
```

**Step 2: Install dependencies**

```bash
conda env create -f environment.yml
conda activate satngp
```

**Step 3: Install `tinycudann`**
Due to CUDA compilation requirements, `tiny-cuda-nn` must be installed manually. Please ensure your `nvcc --version` matches the PyTorch CUDA version, then run:
```bash
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

---

## 2. Dataset Preparation

The original images originate from the [DFC2019 dataset](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019). For convenience, we use the pre-processed version provided by the [Sat-NeRF](https://github.com/centreborelli/satnerf/releases/download/EarthVision2022/dataset.zip) repository.

Run the following commands to download and extract the dataset:

```bash
mkdir data
cd data
wget https://github.com/centreborelli/satnerf/releases/download/EarthVision2022/dataset.zip
unzip dataset.zip -d dataset
cd ..
```

To simplify the execution commands for training and evaluation, please set the path to your dataset and desired output directory as environment variables:

```bash
export DATA_DIR=$(pwd)/data/dataset
export OUT_DIR=/path/to/your/sathg-output
```

---

## 3. Training

Use `main.py` to train the Sat-HG model. The following command provides an example of training on the `JAX_260` AOI with full features enabled (appearance embedding, radiometric normalization, distortion loss, etc.).

```bash
python3 main.py --model sat-nerf-ngp --exp_name --gpu_id 0 --root_dir $DATA_DIR/root_dir/crops_rpcs_ba_v2/JAX_260 --img_dir $DATA_DIR/DFC2019/Track3-RGB-crops/JAX_260 --cache_dir $DATA_DIR/DFC2019/cache_dir/JAX_260_ngp --gt_dir $DATA_DIR/DFC2019/Track3-Truth --logs_dir $OUT_DIR/logs --ckpts_dir $OUT_DIR/ckpts --lr 0.005 --chunk 4096 --max_train_steps 100000 --use_smoothness_loss --use_appearance_embedding --use_distortion_loss --radiometric_normalization --img_downscale 2.0
```

> **Note**: A timestamp will be automatically prepended to your `--exp_name` (e.g., `2026-05-11_15-24-41_JAX_260_lr0.005`). You will need this full identifier for the evaluation step.

---

## 4. Evaluation and DSM Generation

Use `eval_sathg.py` to evaluate the trained model, compute quantitative metrics (PSNR, SSIM), and extract the Digital Surface Model (DSM). 

Please replace `exp_name` with your actual generated experiment ID, and `epochID` with your target epoch number.

```bash
python3 eval_sathg.py exp_name $OUT_DIR/logs $OUT_DIR/eval_results epochID test --checkpoints_dir $OUT_DIR/ckpts --root_dir $DATA_DIR/root_dir/crops_rpcs_ba_v2/JAX_260 --img_dir $DATA_DIR/DFC2019/Track3-RGB-crops/JAX_260 --gt_dir $DATA_DIR/DFC2019/Track3-Truth
```
*Expected Output: The script will load the specified checkpoint, render the images/depth maps in chunks without OOM, save the outputs to `$OUT_DIR/eval_results`, and print the final evaluation metrics.*

---

## Acknowledgements

This project is built upon the foundational architecture of the official [Sat-NeRF](https://github.com/centreborelli/sat-nerf) repository. We sincerely thank the original authors for their excellent open-source work. 

Our main structural modifications upon their codebase include:
1. Integration of `tiny-cuda-nn` for **HashGrid representation**.
2. Re-engineering of the ECEF-to-UTM projection pipeline using modern `pyproj` APIs.
3. Refactoring of the chunk-based volume rendering logic to eliminate Out-Of-Memory (OOM) issues during large-scale evaluation.
