from gymnasium.envs.registration import register

register(
    id='Quadrotor-v0',
    entry_point='environments.quadrotor_env:QuadrotorEnv',
)

register(
    id='Quadrotor-Fixed-v0',
    entry_point='environments.quadrotor_fixed_env:QuadrotorFixedEnv',
)

register(
    id='Quadrotor-Perturbed-v0',
    entry_point='environments.quadrotor_perturbed_env:QuadrotorPerturbedEnv',
)

register(
    id='Quadrotor-Perturbed-Lissajous-v0',
    entry_point='environments.quadrotor_perturbed_lissajous_env:QuadrotorPerturbedLissajousEnv',
)

# Dictionary of all environment classes that require custom command line arguments.
CUSTOM_ENV_CLASSES = {
    'Quadrotor-v0': 'environments.quadrotor_env:QuadrotorEnv',
    'Quadrotor-Fixed-v0': 'environments.quadrotor_fixed_env:QuadrotorFixedEnv',
    'Quadrotor-Perturbed-v0': 'environments.quadrotor_perturbed_env:QuadrotorPerturbedEnv',
    'Quadrotor-Perturbed-Lissajous-v0': 'environments.quadrotor_perturbed_lissajous_env:QuadrotorPerturbedLissajousEnv',

}