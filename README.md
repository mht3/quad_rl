# quad_rl
Quadrotor gymnasium environments and baseline RL implementations.

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
python main.py --env_id Quadrotor-Fixed-v0 --algorithm PPO --policy_net 64 64 --value_net 64 64 --seed 42 
```

## Model Playback

```bash
python main.py --env_id Quadrotor-Fixed-v0 --algorithm PPO --policy_net 64 64 --value_net 64 64 --seed 42 --test --render
```