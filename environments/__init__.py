from gymnasium.envs.registration import register

register(
    id='Quadrotor-v0',
    entry_point='environments.quadrotor_env:QuadrotorEnv',
)

register(
    id='Quadrotor-Fixed-v0',
    entry_point='environments.quadrotor_fixed_env:QuadrotorFixedEnv',
)

# Dictionary of all environment classes that require custom command line arguments.
CUSTOM_ENV_CLASSES = {
    'Quadrotor-v0': 'environments.quadrotor_env:QuadrotorEnv',
    'Quadrotor-Fixed-v0': 'environments.quadrotor_fixed_env:QuadrotorFixedEnv',
}