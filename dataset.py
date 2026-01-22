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

    def __init__(self, env, policy, N=10000, sensor_measurements=False):
        self.env = env
        # numer of different trajectories to collect
        self.N = N
        # lqg policy
        self.policy = policy
        self.sensor_measurements = sensor_measurements

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
            if self.sensor_measurements:
                current_reference = self.env.get_current_reference()
                _obs = self.env.quadrotor.get_sensor_measurements()
                model_input = current_reference[:4] - _obs
            else:
                model_input = obs

            # inputs -> actions
            trajectory.append((model_input, act))

            obs, reward, terminate, truncate, info = self.env.step(act)

        return trajectory

    def save(self, trajectories, filename='trajectories.npz'):
        trajectories_array = np.array(trajectories, dtype=object)        
        np.savez_compressed(filename, 
                            trajectories=trajectories_array,
                            sensor_measurements=self.sensor_measurements,
                            N=len(trajectories))

def parse_args():
    parser = argparse.ArgumentParser(description='Generate quadrotor trajectory dataset')
    parser.add_argument('--num_trajectories', type=int, default=1000,
                        help='Number of trajectories to collect (default: 10000)')
    parser.add_argument('--sensor_measurements', action='store_true', default=False,
                        help='Whether to use sensors to measure the state. (default: False)')
    parser.add_argument("--perturbation_std", type=float, default=0.05, required=True,
                        help='standard deviation for waypoitns to perturb')
    parser.add_argument("--fixed_perturbation_seed", type=int, default=18, required=True,
                        help='help')
    parser.add_argument("--time_per_waypoint", type=float, default=0.1562505, required=True,
                        help='standard deviation for waypoitns to perturb')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file. If None, uses default path based on log name.')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    num_trajectories = args.num_trajectories
    sensor_measurements = args.sensor_measurements

    ### environment setup
    env = gym.make('Quadrotor-Fixed-v0', perturbation_std=args.perturbation_std, fixed_perturbation_seed=args.fixed_perturbation_seed, time_per_waypoint=args.time_per_waypoint)
    env = env.unwrapped


    print("Loading model...", end=' ')
    controller = algorithms.PPOTrainer.load(args.model_path)
    print("Done.")
    print('##### Data Collection ######')

    # save path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(cur_dir, 'datasets')
    os.makedirs(dataset_dir, exist_ok=True)

    if sensor_measurements:
        input_name = 'obs'
        output_name = 'action'
    else:
        input_name = 'state'
        output_name = 'action' 

    save_path = os.path.join(dataset_dir, 'quadrotor_fixed_pert_std_{}_in_{}_out_{}_{}t.npz'.format(args.perturbation_std, input_name, output_name, num_trajectories))
    dataset = TrajectoryCollector(env, policy=controller, N=num_trajectories, sensor_measurements=sensor_measurements)
    print('Building Dataset...')
    trajectories = dataset.build()
    # make sure there are no crashes
    print('Trajectory lengths: {}'.format(len(trajectories[0])))
    for i, traj in enumerate(trajectories):
        assert len(traj) == len(trajectories[0]), f'traj {i}: {len(traj)} != {len(trajectories[0])}'

    print('Saving dataset of {} trajectories...'.format(len(trajectories)))
    dataset.save(trajectories, filename=save_path)

    print("Input Shape: {}".format(len(trajectories[0][0][0])))
    print("Output Shape: {}".format(len(trajectories[0][0][1])))
