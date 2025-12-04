from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import environments
import algorithms
import utils
from scipy import linalg

def plot_results(data, env, filename):
    """
    Plot flight results:
    - Row 1: x, y, z (m) vs desired; scatter noisy measurements
    - Row 2: yaw (rad) vs desired; scatter noisy measurements
    - Row 3: Control inputs: tau_x, tau_y, tau_z (N·m), f_z (N)
    Saves figure to 'models/time_results.png' under cur_dir.
    """
    t = data['t']

    # fixed colors
    colors = {
        'x': 'C0',
        'y': 'C1',
        'z': 'C2',
        'yaw': 'C2',
        'pitch': 'C1',
        'roll' : 'C0'
    }

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12), sharex=True)

    # Row 1: positions
    ax1 = axes[0]
    for var in ['x', 'y', 'z']:
        c = colors[var]
        ax1.plot(t, data[var], color=c, label=rf'${var}$')
        ax1.plot(t, data[f"{var}_des"], '--', color=c, label=rf'${var}_{{des}}$')

    # ax1.set_ylim(-1.5, 1.5)
    ax1.set_ylabel('Position (m)', fontsize=16)
    ax1.legend(loc='upper right', fontsize=14)
    ax1.grid(True)

    # Row 2: roll, pitch, yaw
    ax2 = axes[1]
    ax2.plot(t, data['yaw'], color=colors['yaw'], label=r'$\psi$')
    ax2.plot(t, data['yaw_des'], '--', color=colors['yaw'], label=r'$\psi_{des}$')

    ax2.plot(t, data['pitch'], color=colors['pitch'], label=r'$\theta$')
    ax2.plot(t, data['roll'], color=colors['roll'], label=r'$\phi$')

    # ax2.set_ylim(-0.3, 0.3)
    ax2.set_ylabel('Euler Angles (rad)', fontsize=16)
    ax2.legend(loc='upper right', fontsize=14)
    ax2.grid(True)

    # Row 3: control inputs
    ax3 = axes[2]
    ax3.plot(t, data['tau_x'], label=r'$\tau_x$ (N·m)')
    ax3.plot(t, data['tau_y'], label=r'$\tau_y$ (N·m)')
    ax3.plot(t, data['tau_z'], label=r'$\tau_z$ (N·m)')
    fz_dev = np.array(data['f_z'])
    ax3.plot(t, fz_dev, label=r'$f_z - m \cdot g$ (N)')
    ax3.set_ylabel('Control Inputs', fontsize=16)
    # ax3.set_ylim(-0.5, 0.5)

    ax3.set_xlabel('time (s)', size=16)
    ax3.legend(loc='upper right', fontsize=14)
    ax3.grid(True)

    fig.tight_layout()

    # save figure
    cur_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(cur_path, 'images', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Plot saved to {save_path}")

def track_data(obs, action, env, model, data):
    if len(data) == 0:
        # initialize
        data['t'] = []
        data['x'] = []
        data['y'] = []
        data['z'] = []
        data['yaw'] = []
        data['roll'] = []
        data['pitch'] = []
        data['v_x'] = []
        data['v_y'] = []
        data['v_z'] = []
        data['w_x'] = []
        data['w_y'] = []
        data['w_z'] = []
        data['x_meas'] = []
        data['y_meas'] = []
        data['z_meas'] = []
        data['yaw_meas'] = []
        data['x_des'] = []
        data['y_des'] = []
        data['z_des'] = []
        data['yaw_des'] = []
        data['tau_x'] = []
        data['tau_y'] = []
        data['tau_z'] = []
        data['f_z'] = []

    data['t'].append(env.t)
    data['x'].append(env.state[0])
    data['y'].append(env.state[1])
    data['z'].append(env.state[2])
    data['yaw'].append(env.state[3])
    data['pitch'].append(env.state[4])
    data['roll'].append(env.state[5])
    data['v_x'].append(env.state[6])
    data['v_y'].append(env.state[7])
    data['v_z'].append(env.state[8])
    data['w_x'].append(env.state[9])
    data['w_y'].append(env.state[10])
    data['w_z'].append(env.state[11])
    # measurements
    data['x_meas'].append(obs[0])
    data['y_meas'].append(obs[1])
    data['z_meas'].append(obs[2])
    data['yaw_meas'].append(obs[3])

    # reference
    reference_pos = env.get_current_reference()
    data['x_des'].append(reference_pos[0])
    data['y_des'].append(reference_pos[1])
    data['z_des'].append(reference_pos[2])
    data['yaw_des'].append(reference_pos[3])

    # convert motor speeds to torques if control_motors=True
    if env.control_motors:
        s = env.unnormalize_action(action) ** 2
        action = linalg.solve(env.M, s)
        
    data['tau_x'].append(action[0])
    data['tau_y'].append(action[1])
    data['tau_z'].append(action[2])
    data['f_z'].append(action[3] - env.m* env.g)
    
if __name__ == '__main__':
    cur_path = os.path.dirname(os.path.realpath(__file__))
    show_gui = False
    perturbation_std = 0.0
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'Quadrotor-Fixed-Perturbed-v0/PPO_pi_512-256-128_vf_512-256-128_s_42_best.zip')
    if show_gui:
        render_mode = 'human'
    else:
        render_mode = None
    
    env = gym.make('Quadrotor-Fixed-Perturbed-v0', perturbation_std=perturbation_std, render_mode=render_mode)
    env = env.unwrapped
    # Load model
    print("Loading model...", end=' ')
    model = algorithms.PPOTrainer.load(model_path)
    print("Done.")

    terminate = False
    truncate = False
    obs,_ = env.reset()
    rew = 0

    data = {}
    while not truncate and not terminate:
        action = utils.model_inference(obs, model)
        if env.time_step % 5 == 0:
            track_data(obs, action, env, model, data)
        obs, rewards, terminate, truncate, info = env.step(action)
        rew += rewards
        if (truncate or terminate) and show_gui:
            # save last frame of quadrotor and save as 'flight_result_perturb_{perturbation_std}.png'
            filename = f'flight_result_perturb_{perturbation_std}.png'
            save_path = os.path.join(cur_path, 'images', filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            env.quadrotor.ax.figure.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Flight result saved to {save_path}")
    
    print("Episode Reward: {}".format(rew))
    filename = f'time_results_perturb_{perturbation_std}.png'

    plot_results(data, env, filename)