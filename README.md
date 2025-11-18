# quad_rl
Quadrotor gymnasium environments and baseline RL implementations.

<img width="388" height="383" alt="image" src="https://github.com/user-attachments/assets/4d2003d6-db3d-4e0d-8fed-a5cfaae3aeec" />

## Getting Started

Clone the environment and change directories. The following uses cloning via ssh:

```bash
git@github.com:mht3/quad_rl.git
cd quad_rl
```

### Environment Setup

Create a new conda environment with Python 3.11.
```bash
conda create -n quad python=3.11
```

Activate the environment.
```sh
conda activate quad
```

Install torch

<details>
<summary>PyTorch on GPU</summary>
<br>
Install a CUDA enabled PyTorch that matches your system architecture.
  
```sh
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```
</details>

<details>
<summary>PyTorch on CPU Only</summary>
<br>
Alternatively, install PyTorch on the CPU.
  
```sh
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
```
</details>

Install the remaining required packages.
```bash
pip install -r requirements.txt
```

## Model Training

```bash
python main.py --env_id Quadrotor-Fixed-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.00005 --policy_net 512 256 128 --value_net 512 256 128
```

## Model Playback

```bash
python main.py --env_id Quadrotor-Fixed-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.00005 --policy_net 512 256 128 --value_net 512 256 128 --test --render
```
