python main.py --env_id Quadrotor-Fixed-Perturbed-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.0001 --policy_net 512 256 128 --value_net 512 256 128 --perturbation_std 0.05
python main.py --env_id Quadrotor-Fixed-Perturbed-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.00005 --policy_net 512 256 128 --value_net 512 256 128 --perturbation_std 0.05

