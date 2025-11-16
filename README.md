# <p align=center> [ICLR] GAGA: Gaussianity-Aware Gaussian Approximation for Efficient 3D Molecular Generation</p>

<div align="center">

</div>

---

>**GAGA:  Gaussianity-Aware Gaussian Approximation for Efficient 3D Molecular Generation** <br>
>ICLR 2026 in Submission<br>

> **Introduction:** *Gaussian Probability Path based Generative Models (GPPGMs) generate data by reversing a stochastic process that progressively corrupts samples with Gaussian noise. Despite state-of-the-art results in 3D molecular generation, their deployment is hindered by the high cost of long generative trajectories, often requiring hundreds to thousands of steps during training and sampling. In this work, we propose a principled method, named GAGA, to improve generation efficiency without sacrificing the training granularity or inference fidelity of GPPGMs. Our key insight is that different data modalities converge to Gaussianity at markedly different rates during the forward process. Based on this observation, we analytically identify a characteristic step at which molecular data attain sufficient Gaussianity, after which the trajectory can be replaced by a closed-form Gaussian approximation. Unlike existing accelerators that coarsen or reformulate trajectories, our approach preserves full-resolution learning dynamics while avoiding redundant transport through distributional states with sufficient Gaussianity. Experiments on 3D molecular generation benchmarks demonstrate that our GAGA achieves substantial improvement on both generation quality and computational efficiency.*
<hr />

# Gaussianity-Aware Gaussian Approximation (GAGA)

This repository contains code and scripts for reproducing the **GAGA**. In this project, we use [EDM](https://arxiv.org/abs/2203.17003) as an example to plug in GAGA algorithm. Because most other 3D molecular generation baselines are constructed on EDM, so it's easy to transfer EDM-GAGA to other baselines-GAGA.

---

## 1. Estimate and Save Variance

Estimate and store the variance of each sample, then compute the dataset-level variance using:

```bash
python Variance_Estimate.py --model_path=outputs/edm_qm9_max_550 --test_name=QM9
```

The program will print the variance across all samples. It takes some time to run it on QM9 and GEOM datasets. So we provide the stats result in step 3.

## 2. Gaussianity Evaluation (KS and MI Tests)

Estimate the **Kolmogorov–Smirnov (KS)** and **Mutual Information (MI)** metrics heuristically, and record the Gaussianity passing state for each timestep:

```bash
# for QM9
python Gaussianity_test.py --model_path outputs/edm_qm9_GA_550 --test_name=QM9

# for GEOM
python Gaussianity_test.py --model_path outputs/edm_geom_drugs_GA_550 --test_name=GEOM

```

Note: the reason why we need the model here is the noising schedule is saved in the model. So this model path can direct to either original EDM weights or GAGA-trained weights.

The Gaussianity test result will be saved to: 'identity_to_gaussian_{test_name}'

## 3. GAGA Timestep Estimation

Estimate the GAGA timestep (`T*`) using the following command:

```bash
python GAGA_step_estimation.py --data_root identity_to_gaussian_{test_name}
```

It will take some time to run the statistical test on the whole dataset in steps 2 and 3. Therefore, we provide the stats of QM9 and GEOM(random sampled 10%) here:

Dataset Variance:

QM9: 0.848./ GEOM: 0.952.

GAGA Timestep:

QM9: 550/ GEOM: 650.

Note: To improve the robustness of statistical test, we guranteed that the GA timestep has consistently passed the statistical test. 

## 4. Training and Sampling with GA-Based Models

We provide an example to plug in the GAGA into EDM for both training and sampling efficiency improvement. 

For training:

### QM9

```bash
python main_qm9.py --n_epochs 1650 --exp_name edm_qm9_GA_550 \
--n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 \
--diffusion_noise_precision 1e-5 --diffusion_steps 1000 \
--diffusion_loss_type l2 --batch_size 256 --nf 256 --n_layers 9 \
--lr 1e-4 --normalize_factors [1,4,10] --test_epochs 500 \
--ema_decay 0.9999 --include_charge=False --dp=True \
--max_t=550 --no_wandb
```

### GEOM

```bash
python main_geom_drugs.py --n_epochs 60 --exp_name edm_geom_drugs_GA_650 \
--n_stability_samples 500 --diffusion_noise_schedule polynomial_2 \
--diffusion_steps 1000 --diffusion_noise_precision 1e-5 \
--diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 4 \
--lr 1e-4 --normalize_factors [1,4,10] --test_epochs 5 \
--ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics \
--visualize_every_batch 10000 --max_t=650 --no_wandb
```

For sampling:

### QM9

```bash
python GAGA_test.py --model_path=outputs/edm_qm9_max_550 \
--n_samples=10000 --sampling_step_size=1 --shortcut=Gaussian \
--test_name=QM9_GA --dataset_var=0.848 --GA_timestep=550 --iteration=3
```

### GEOM

```bash
python GAGA_test.py --model_path=outputs/edm_geom_max_650 \
--n_samples=10000 --sampling_step_size=1 --shortcut=Gaussian \
--test_name=GEOM_GA --dataset_var=0.952 --GA_timestep=650 --iteration=3
```
