python main.py --env_id Quadrotor-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 4092 --batch_size 256 --lr 0.00009 --policy_net 512 256 128 --value_net 512 256 128
python main.py --env_id Quadrotor-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.0001 --policy_net 512 256 128 --value_net 512 256 128
python main.py --env_id Quadrotor-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 4092 --batch_size 128 --lr 0.0001 --policy_net 512 256 128 --value_net 512 256 128

# python main.py --env_id Quadrotor-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.00005 --policy_net 512 256 128 --value_net 512 256 128
# python main.py --env_id Quadrotor-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.00005 --policy_net 256 256 256 --value_net 256 256 256 

# python main.py --env_id Quadrotor-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.00005 --policy_net 512 256 128 --value_net 512 256 128 --history_len 5 --flatten_observation
# python main.py --env_id Quadrotor-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.00005 --policy_net 256 256 256 --value_net 256 256 256 --history_len 5 --flatten_observation
# python main.py --env_id Quadrotor-v0 --algorithm PPO --seed 42 -t 10000000 --n_steps 3072 --batch_size 256 --lr 0.0001 --policy_net 512 256 128 --value_net 512 256 128 --history_len 5 --flatten_observation
