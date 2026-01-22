python main.py \
    --env_id Quadrotor-Fixed-v0 \
    --algorithm PPO \
    --seed 42 \
    -t 75000000 \
    --n_steps 6148 \
    --batch_size 512 \
    --ent_coef 0.0 \
    --lr 0.00002 \
    --gamma 0.99 \
    --policy_net 512 256 128 128 \
    --value_net 512 256 128 128 \
    --perturbation_std 0.2 \
    --fixed_perturbation_seed 18 \
    --time_per_waypoint 0.234375

