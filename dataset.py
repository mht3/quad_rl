import numpy as np
import gymnasium as gym
import sys, os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import environments
import algorithms

'''
Dataset collection tool for quadrotor.
Example usage:
    python dataset.py --num_trajectories 1000

This will save an .npz file named `quadrotor_fixed_pert_std_0.05_in_obs_wp_out_action_1000t` of 1000 trajectories to the datasets folder.

The inputs in this case are observations (noisy x, y, z, yaw measurements).
The outputs are the actions (size 4) containing the quadrotor motor speeds.
'''
class TrajectoryCollector():

    def __init__(self, env, policy, N=10000, output_is_state=False,
                 include_actions_in_input=False, include_state_in_output=False):
        self.env = env
        # numer of different trajectories to collect
        self.N = N
        # lqg policy
        self.policy = policy
        # dataset uses controller actions as output or full state
        self.output_is_state = output_is_state
        # whether or not to include the actions (with partial state) in the input if the output is the full state
        self.include_actions_in_input = include_actions_in_input
        self.include_state_in_output = include_state_in_output

    def get_action(self, x):
        '''
        Take in single observation x to get action 
        '''
        with torch.inference_mode():
            action, _ = self.policy.predict(x)
        return action

    def build(self):
        trajectories = []
        for i in tqdm(range(self.N)):
            # reset environment for new trajectory
            obs, _ = self.env.reset()
            trajectory = self.get_trajectory(obs)
            trajectories.append(trajectory)

        return trajectories

    def get_trajectory(self, obs):
        terminate = False
        truncate = False

        trajectory = []
        act = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        while not truncate and not terminate:
            prev_act = act.copy()
            act = self.get_action(obs)
            
            if self.output_is_state:
                # full state is 12 dims (quadrotor state)
                full_state = self.env.state
                # only include sensor measurements (partial state) for dataset (waypoints not needed for state reconstruction)
                _obs = obs[0:4] 
                if self.include_actions_in_input:
                    # include o_t and a_{t-1} to reconstruct s_t
                    model_input = np.concatenate([_obs, prev_act])
                else:
                    model_input = _obs
                trajectory.append((model_input, full_state))
            else:
                sensor_measurements = obs[0:4]
                # ref x, y, z, psi, vx, vy, vz
                waypoints = obs[-7:]
                _obs = np.concatenate([sensor_measurements, waypoints])
                if self.include_actions_in_input:
                    # include o_t and a_{t-1} to reconstruct s_t
                    model_input = np.concatenate([_obs, prev_act])
                else:
                    model_input = _obs

                if self.include_state_in_output:
                    # observations + waypoints -> actions + state
                    full_state = self.env.state
                    _out = np.concatenate([act, full_state])
                    trajectory.append((model_input, _out))
                else:
                    # observations + waypoints -> actions
                    trajectory.append((model_input, act))

            obs, reward, terminate, truncate, info = self.env.step(act)

        return trajectory

    def save(self, trajectories, filename='trajectories.npz'):
        trajectories_array = np.array(trajectories, dtype=object)        
        np.savez_compressed(filename, 
                            trajectories=trajectories_array,
                            output_is_state=self.output_is_state,
                            include_actions_in_input=self.include_actions_in_input,
                            include_state_in_output=self.include_state_in_output,
                            N=len(trajectories))

def parse_args():
    parser = argparse.ArgumentParser(description='Generate quadrotor trajectory dataset')
    parser.add_argument('--num_trajectories', type=int, default=1000,
                        help='Number of trajectories to collect (default: 10000)')
    parser.add_argument('--output_is_state', action='store_true', default=False,
                        help='Whether output should be full state (default: False)')
    parser.add_argument('--include_actions_in_input', action='store_true', default=False,
                        help='Whether to include actions in input when output is state (default: False)')
    parser.add_argument('--include_state_in_output', action='store_true', default=False,
                        help='Whether to include state in output along with actions (when --output_is_state is false). Gives Input as obs+waypoints and outputs action + state (default: False)')
    parser.add_argument("--perturbation_std", type=float, default=0.05, required=True,
                        help='help')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file. If None, uses default path based on log name.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    num_trajectories = args.num_trajectories
    output_is_state = args.output_is_state
    include_actions_in_input = args.include_actions_in_input
    include_state_in_output = args.include_state_in_output

    if output_is_state and include_state_in_output:
        raise ValueError("`include_state_in_output` only valid for action prediction setting.")

    ### environment setup
    env = gym.make('Quadrotor-Fixed-Perturbed-v0', perturbation_std=args.perturbation_std)
    env = env.unwrapped


    print("Loading model...", end=' ')
    controller = algorithms.PPOTrainer.load(args.model_path)
    print("Done.")
    print('##### Data Collection ######')

    # save path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(cur_dir, 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)

    if output_is_state:
        output_name = 'state'
        if include_actions_in_input:
            input_name = 'obs_act' 
        else:
            input_name = 'obs'
    else:
        # observation (4) and target waypoint (7) included in input (target waypoint needed to predict future actions)
        input_name = 'obs_wp'
        output_name = 'action_state' if include_state_in_output else 'action'

    save_path = os.path.join(dataset_dir, 'quadrotor_fixed_pert_std_{}_in_{}_out_{}_{}t.npz'.format(args.perturbation_std, input_name, output_name, num_trajectories))
    dataset = TrajectoryCollector(env, policy=controller, N=num_trajectories, output_is_state=output_is_state,
                                  include_actions_in_input=include_actions_in_input, include_state_in_output=include_state_in_output)
    print('Building Dataset...')
    trajectories = dataset.build()
    # make sure there are no crashes
    print('Trajectory lengths: {}'.format(len(trajectories[0])))
    for traj in trajectories:
        assert len(traj) == len(trajectories[0])

    print('Saving dataset of {} trajectories...'.format(len(trajectories)))
    dataset.save(trajectories, filename=save_path)

    print("Input Shape: {}".format(len(trajectories[0][0][0])))
    print("Output Shape: {}".format(len(trajectories[0][0][1])))
