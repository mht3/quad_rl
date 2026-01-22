python main.py \
    --env_id Quadrotor-Fixed-Perturbed-v0 \
    --algorithm PPO \
    --seed 42 \
    -t 20000000 \
    --n_steps 5120 \
    --batch_size 512 \
    --lr 0.00002 \
    --gamma 0.99 \
    --ent_coef 0.0 \
    --vf_coef 0.5 \
    --policy_net 512 256 128 \
    --value_net 512 256 128 \
    --perturbation_std 0.05

