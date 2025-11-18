import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from .quadrotor_utils import Quadrotor, TrajectoryGenerator, run_single_episode
from scipy import linalg
import time

class QuadrotorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, waypoints=None, total_time=None, render_mode=None, control_motors=True,
                 normalized_actions=True, fully_observable=True, boundary_length=5,
                 time_per_waypoint=0.15625, add_takeoff_waypoint=False):
        '''
        Initializes the quadrotor environment. More complicated, longer lissajous curves with many twists and turns. Each trajectory is 18 seconds, or 1800 steps
        Waypoints:
            2D list or numpy array of goal positional coordinates for the drone.
        total_time:
            total time for drone to fly through waypoints. Used for creating smooth trajectories. Defaults to 4 seconds per waypoint.
        render_mode:
            Optional argument to display quadrotor path. Defaults to None
        control_motors:
            Flag to switch control inputs from net torques and forces to the angular velocity of each rotor. Defaults to false.
        normalized_actions:
            Flag for if actions passed to the environment are normalized. (Either motor commands or net torques and z-force). Defaults to True.
        fully_observable:
            Flag for if the state is fully observable. If False the state is assumed to be noisy measurements of (x, y, z, yaw).
        boundary_length:
            Boundary side length for arena centered at the origin. Defaults to 5 meters.
        default_time_per_waypoint: float
            Time per waypoint for quadrotor.
        add_takeoff_waypoint: bool
            If True, automatically add takeoff waypoint at index 0. If False, use waypoints as-is.
        '''
        super().__init__()
        # simulation params
        self.add_takeoff_waypoint = add_takeoff_waypoint
        self.total_time = total_time
        if waypoints is None:
            # will generate randomized Lissajous path
            num_waypoints = 64
        else:
            num_waypoints = len(waypoints)
        if self.total_time is None:
            self.total_time = num_waypoints * time_per_waypoint
        self.max_runtime = 1.2 * self.total_time
        self.dt = 0.01
        self.max_time_steps = int(self.max_runtime / self.dt)
        self.max_reference_time_steps = int(self.total_time / self.dt)
        self.control_motors = control_motors
        # gravity (m/s^2)
        self.g = 9.81
        # mass of drone kg
        self.m = 0.5
        # moments of intertia (kg m^2)
        self.Jx = 0.0023
        self.Jy = 0.0023
        self.Jz = 0.0040
        # distance from center of drone to each rotor
        self.l = 0.175

        # Parameters that govern applied force and torque
        self.kF = 7e-6
        self.kM = 1e-7
        # rotor speed bounds (rad / s)
        self.min_spin_rate = 100 
        self.max_spin_rate = 900
        self.s_min = self.min_spin_rate**2
        self.s_max = self.max_spin_rate**2
        # relate spin rates to forces and torques
        self.M = linalg.inv(np.array([[0., 0., self.kF * self.l, -self.kF * self.l],
                                      [-self.kF * self.l, self.kF * self.l, 0., 0.],
                                      [-self.kM, -self.kM, self.kM, self.kM],
                                      [self.kF, self.kF, self.kF, self.kF]]))

        # side length for cube boundaries of arena
        self.boundary_length = boundary_length

        self.render_mode = render_mode
        if self.render_mode == "human":
            show_animation = True
        else:
            show_animation = False


        # setup quadrotor object
        self.quadrotor = Quadrotor(self.m, self.Jx, self.Jy, self.Jz, self.l, self.g, boundary_length=boundary_length, show_animation=show_animation)

        # f(x, u) = x_dot 
        self.f = self.quadrotor.get_system_dynamics()

        self.fully_observable = fully_observable

        # full state space is position, orientation, linear velocity, and angular velocity and reference (x, y, z, yaw, vx, vy, vz)
        state_high = np.array([boundary_length/2, boundary_length/2, boundary_length, np.pi, np.pi, np.pi, 5, 5, 5, np.pi, np.pi, np.pi,
                                boundary_length/2, boundary_length/2, boundary_length, np.pi, 5, 5, 5], dtype=np.float32)
        state_low = np.array([-boundary_length/2, -boundary_length/2, 0, -np.pi, -np.pi, -np.pi, -5, -5, -5, -np.pi, -np.pi, -np.pi,
                            -boundary_length/2, -boundary_length/2, 0, -np.pi, -5, -5, -5], dtype=np.float32)

        self.state_space = Box(low=state_low, high=state_high, dtype=np.float32)

        if self.fully_observable:
            self.observation_space = self.state_space
        else:
            # partial observable case
            # observation space is noisy measurments of position, yaw and reference (x, y, z, yaw)
            obs_high = np.array([boundary_length/2, boundary_length/2, boundary_length, np.pi,
                                    boundary_length/2, boundary_length/2, boundary_length, np.pi,], dtype=np.float32)
            obs_low = np.array([-boundary_length/2, -boundary_length/2, 0, -np.pi,
                                -boundary_length/2, -boundary_length/2, 0, -np.pi,], dtype=np.float32)
            self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.normalized_actions = normalized_actions
        if self.control_motors:
            #  motor[0]: front (+z spin)
            #  motor[1]: rear (+z spin)
            #  motor[2]: left (-z spin) (negative already accounted for in model)
            #  motor[3]: right (-z spin)
            if normalized_actions:
                # define unnormalization function 
                self.unnormalize_action = lambda action: (action + 1) * 0.5 * (self.max_spin_rate - self.min_spin_rate) + self.min_spin_rate 
                self.action_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            else:
                # do nothing for unnormalization
                self.unnormalize_action = lambda action : action
                self.action_space = Box(low=self.min_spin_rate, high=self.max_spin_rate,shape=(4,), dtype=np.float32)
        else:
            # action space is net torques and thrust in z direction. (tau_x, tau_y, tau_z, f_z)
            tau_max = 2.
            f_max = 3 * self.m * self.g
            action_low = np.array([-tau_max, -tau_max, -tau_max, 0], dtype=np.float32)
            action_high = np.array([tau_max, tau_max, tau_max, f_max], dtype=np.float32)
            self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)
            self.unnormalize_action = lambda action : action

        # will generate self.waypoints 
        self._process_waypoints(waypoints)

        # get smooth reference trajectory to follow
        time_per_waypoint = self.total_time / (len(self.waypoints) - 1)
        self.reference = self.get_reference_trajectory(self.waypoints, time_per_waypoint)
        # Update max_reference_time_steps to match actual reference trajectory length
        self.max_reference_time_steps = self.reference.shape[1]
        # plot waypoints and reference if needed
        if self.render_mode == 'human':
            self.quadrotor.set_waypoints(self.waypoints)
            self.quadrotor.set_reference_trajectory(self.reference)
        # info keywords for logging.
        self.info_keywords = ('position_rew', 'angle_rew', 'velocity_rew', 'survival_rew', 'action_reg_rew')

        # initialize and reset env
        self.reset()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--no_control_motors', action='store_false', dest='control_motors', default=True)
        parser.add_argument('--no_normalized_actions', action='store_false', dest='normalized_actions', default=True)
        parser.add_argument('--partially_observable', action='store_false', dest='fully_observable', default=True)
        parser.add_argument('--boundary_length', type=int, default=5)
        parser.add_argument('--total_time', type=float, default=None)
        parser.add_argument('--time_per_waypoint', type=float, default=0.15625)

    
    @staticmethod
    def get_env_kwargs(args):
        kwargs = {'control_motors': args.control_motors,
                  'normalized_actions': args.normalized_actions,
                  'fully_observable': args.fully_observable,
                  'boundary_length': args.boundary_length,
                  'total_time': args.total_time,
                  'time_per_waypoint': args.time_per_waypoint
                  }

        return kwargs

    def generate_lissajous_waypoints(self, n_waypoints=64):
        """
        Generate waypoints following a 3D Lissajous curve pattern.
        https://en.wikipedia.org/wiki/Lissajous_curve

        Lissajous parameterization:
        x(t) = alpha*sin(t)
        y(t) = beta*sin(n*t + phi)
        z(t) = gamma*sin(m*t + psi) + z_offset
        
        Args:
            n_waypoints: Number of waypoints to generate
            
        Returns:
            numpy array of [x, y, z, yaw] waypoints
        """
        # y oscillations
        n = np.random.choice([2, 3, 4])
        # z oscillations
        m = np.random.choice([1, 2, 3])

        z_offset = np.random.uniform(2.5, 3.25)
        phi = np.random.uniform(0, np.pi)
        psi = np.random.uniform(0, np.pi)
        
        alpha = 2
        beta = 2
        gamma = 1.1
        
        waypoints = np.zeros((n_waypoints, 4))
        
        for i in range(n_waypoints):
            t = 2 * np.pi * i / (n_waypoints - 1)
            
            x = alpha * np.sin(t)
            y = beta * np.sin(n * t + phi)
            z = gamma * np.sin(m * t + psi) + z_offset
            
            waypoints[i, 0] = x
            waypoints[i, 1] = y
            waypoints[i, 2] = z
            waypoints[i, 3] = 0.0
            
        return waypoints

    def _process_waypoints(self, waypoints=None):

        if waypoints is None:
            self.randomize_waypoints = True
            waypoints = self.generate_lissajous_waypoints()
        else:
            self.randomize_waypoints = False

        # Sanity check for data type of waypoints 
        if not isinstance(waypoints, (list, np.ndarray)):
            raise ValueError('`waypoints` must be either `np.array` or `list` but found `{}`'.format(type(waypoints)))

        # ensure waypoints are x, y, z, yaw format
        if self.add_takeoff_waypoint:
            # starting position (randomly initialized) is first waypoint     
            num_waypoints = len(waypoints) + 1
            self.waypoints = np.zeros(shape=(num_waypoints, 4))
            passed_waypoints = np.array(waypoints, dtype=np.float32)
            for i in range(len(passed_waypoints)):
                waypoint = passed_waypoints[i]
                for j in range(len(waypoint)):
                    self.waypoints[i+1, j] = waypoint[j]

            # randomly initialize starting state on ground below first waypoint
            self.initialize_state()
            self.waypoints[0] = self.state[0:4]
        else:
            # use waypoints as-is, no takeoff waypoint added
            num_waypoints = len(waypoints)
            self.waypoints = np.zeros(shape=(num_waypoints, 4))
            passed_waypoints = np.array(waypoints, dtype=np.float32)
            for i in range(len(passed_waypoints)):
                waypoint = passed_waypoints[i]
                for j in range(len(waypoint)):
                    self.waypoints[i, j] = waypoint[j]

            # initialize starting state at first waypoint
            self.initialize_state_at_waypoint(self.waypoints[0])

    def get_quadrotor(self):
        return self.quadrotor

    def update_reference_trajectory(self):
        if self.add_takeoff_waypoint:
            self.waypoints[0] = self.state[0:4]
            # initial desired yaw needs to be 0 (may be different from state).t he environment randomizes the initial state, not the yaw waypoint.
            self.waypoints[0, 3] = 0
        # Number of trajectory segments = Number of waypoints - 1
        time_per_waypoint = self.total_time / (len(self.waypoints) - 1)
        if self.add_takeoff_waypoint:
            end_step = int(time_per_waypoint / self.dt)
            # trajectory from start point to first waypoint
            start_trajectory = self.get_reference_trajectory(self.waypoints[:2], time_per_waypoint)
            # update reference trajectory from start to first waypoint
            self.reference[:, 0:end_step] = start_trajectory
        if self.randomize_waypoints:
            # regenerate entire reference trajectory for new Lissajous path
            self.reference = self.get_reference_trajectory(self.waypoints, time_per_waypoint)
        # plot waypoints and reference if needed
        if self.render_mode == 'human':
            self.quadrotor.set_waypoints(self.waypoints)
            self.quadrotor.set_reference_trajectory(self.reference)

    def get_reference_trajectory(self, waypoints, time_per_waypoint):
        '''
        get the reference trajectory (position and velocities, both linear and angular) with a quintic polynomial.
        '''
        num_waypoints = len(waypoints)
        # N - 1 trajectories for N waypoints
        self.x_coeffs = np.empty(shape=(num_waypoints - 1, 6))
        self.y_coeffs = np.empty(shape=(num_waypoints - 1, 6))
        self.z_coeffs = np.empty(shape=(num_waypoints - 1, 6))
        for i in range(num_waypoints - 1):
            next_idx = (i + 1)
            traj = TrajectoryGenerator(waypoints[i], waypoints[next_idx], time_per_waypoint)
            traj.solve()
            self.x_coeffs[i] = traj.x_c.flatten()
            self.y_coeffs[i] = traj.y_c.flatten()
            self.z_coeffs[i] = traj.z_c.flatten()

        # get datapoints for reference trajectory at each state
        time_step_per_waypoint = int(time_per_waypoint / self.dt)
        total_time_steps = time_step_per_waypoint * (num_waypoints - 1)
        reference = np.empty(shape=(7, total_time_steps))
        time_step = 0
        for i in range(num_waypoints - 1):
            time = 0
            for _ in range(time_step_per_waypoint):
                # get x, y, z reference from smooth planner
                ref_x_pos = TrajectoryGenerator.calculate_position(self.x_coeffs[i], time)
                ref_y_pos = TrajectoryGenerator.calculate_position(self.y_coeffs[i], time)
                ref_z_pos = TrajectoryGenerator.calculate_position(self.z_coeffs[i], time)
                # get velocities
                ref_x_vel = TrajectoryGenerator.calculate_velocity(self.x_coeffs[i], time)
                ref_y_vel = TrajectoryGenerator.calculate_velocity(self.y_coeffs[i], time)
                ref_z_vel = TrajectoryGenerator.calculate_velocity(self.z_coeffs[i], time)
                # assume yaw is constant for the current waypoint
                ref_yaw = waypoints[i, 3]
                # update reference trajectory 
                reference[0, time_step] = ref_x_pos
                reference[1, time_step] = ref_y_pos
                reference[2, time_step] = ref_z_pos
                reference[3, time_step] = ref_yaw
                reference[4, time_step] = ref_x_vel
                reference[5, time_step] = ref_y_vel
                reference[6, time_step] = ref_z_vel

                time += self.dt
                time_step += 1

        return reference

    def enforce_motor_limits(self, u):
        '''
        Clips input based upon motor limits
        u: desired input - tau_x_des, tau_y_des, tau_z_des, f_z_des
        '''
        if self.control_motors:
            # if we are controlling motors, the desired input is actually motor speeds.
            s = np.clip(u**2, self.s_min, self.s_max)
        else:
            # compute and bound squared spin rates
            # M @ u gives the squared spin rates for each rotor (array of size 4).
            s = np.clip(self.M @ u, self.s_min, self.s_max)
        # recompute inputs
        u = linalg.solve(self.M, s)
        return u

    def initialize_state(self):
        # get x, y pos of first waypoint (excluding starting point)
        first_waypoint = self.waypoints[1, :]
        
        out_of_bounds = True
        # starting square is 1/8 of total arena size
        factor = 8
        # reduce area until random starting point is not out of bounds (usually this iterates once)
        reduction_factor = 2
        starting_square_size = self.boundary_length / factor
        xy_pos = first_waypoint[0:2]
        while out_of_bounds:
            xy_pos = first_waypoint[0:2] + np.random.uniform(low=-starting_square_size, high=starting_square_size)
            out_of_bounds = self._is_out_of_bounds(xy_pos[0], xy_pos[1], 0)
            # reduce starting square size
            starting_square_size = starting_square_size / reduction_factor
            if out_of_bounds:
                print("Random starting coordinate out of bounds. Retrying with square size of {}.".format(starting_square_size))

        self.state = np.zeros(12)

        self.state[0:2] = xy_pos
        # randomize yaw around 0 
        self.state[3] = np.random.uniform(low=-np.pi/32, high=np.pi/32, size=1)

        # create visited array for waypoints and set first to true
        self.visited_waypoints = np.zeros(self.waypoints.shape[0], dtype=bool)
        self.visited_waypoints[0] = True

    def initialize_state_at_waypoint(self, waypoint):
        """
        Initialize state at a specific waypoint (used when add_takeoff_waypoint=False)
        """
        self.state = np.zeros(12)
        self.state[0:3] = waypoint[0:3]  # x, y, z position
        if len(waypoint) > 3:
            self.state[3] = waypoint[3]  # yaw
        
        self.visited_waypoints = np.zeros(self.waypoints.shape[0], dtype=bool)
        self.visited_waypoints[0] = True

    def angular_error(self, x, x_ref):
        '''
        Returns the angular error between a vector of angles and their references.
        Angles are assumed to be in the range [-pi, pi]
        '''
        angle_difference = (x - x_ref + np.pi) % (2 * np.pi) - np.pi
        return np.linalg.norm(angle_difference)
    def _reward(self, X, u):
        current_position = X[0:3]
        X_ref = np.zeros(12)
        # penalize based on closest reference point and angular error        
        nearest_ref =  self.get_nearest_reference(current_position, nearby_timesteps=100)
        X_ref[0:4] = nearest_ref[0:4]
        X_ref[6:9] = nearest_ref[4:]

        
        X_err = X - X_ref
        distance = np.linalg.norm(X_err[:3])

        angular_err = self.angular_error(X[3:6], X_ref[3:6])
        
        position_std = 0.5 
        angular_std = 0.5 
        velocity_std = 1.0
        
        position_rew = 1.0 * np.exp(-distance / position_std**2)
        angular_rew = 0.15 * np.exp(-angular_err / angular_std**2)
        
        # velocity reward term (used even for partially observable env)
        velocity_error_magnitude = np.linalg.norm(X_err[6:9])
        velocity_rew = 0.1 * np.exp(-velocity_error_magnitude / velocity_std**2)

        reward = position_rew + angular_rew + velocity_rew

        rew_info = {'position_rew': position_rew,
                    'angle_rew': angular_rew,
                    'velocity_rew': velocity_rew}
        if not self._is_out_of_bounds(*current_position):
            survival_rew = 0.01
            reward += survival_rew
            rew_info['survival_rew'] = survival_rew

        # ensure smooth actions between timesteps
        if self.u_prev is not None:
            action_penalty = np.linalg.norm(u - self.u_prev)
            action_reg_std = 0.5
            action_reg_rew = 0.01 * np.exp(-action_penalty / action_reg_std**2)
            reward += action_reg_rew
            rew_info['action_reg_rew'] = action_reg_rew
        self.u_prev = u
                
        return reward, rew_info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_waypoints:
            # regenerate entire Lissajous path with new random parameters
            new_waypoints = self.generate_lissajous_waypoints()
            self.waypoints = new_waypoints

        if self.add_takeoff_waypoint:
            # randomize starting state
            self.initialize_state()
        else:
            self.initialize_state_at_waypoint(self.waypoints[0])

        # get updated smooth trajectory 
        self.update_reference_trajectory()
        # Update max_reference_time_steps to match actual reference trajectory length
        self.max_reference_time_steps = self.reference.shape[1]

        self.t = 0
        self.time_step = 0
        if self.add_takeoff_waypoint:
            self.target_waypoint_idx = 1  # Skip takeoff waypoint, start with first real waypoint
        else:
            self.target_waypoint_idx = 0  # Start with first waypoint directly
        self.start_time = time.time()
        self.quadrotor.initialize_flight_tracking(x=self.state[0], y=self.state[1], z=self.state[2],
                                                  yaw=self.state[3], pitch=self.state[4], roll=self.state[5])
        # init prev action
        self.u_prev = None

        obs = self._get_obs()
        # Initialize episode reward accumulators
        self.episode_position_rew = 0.0
        self.episode_angle_rew = 0.0
        self.episode_velocity_rew = 0.0
        self.episode_survival_rew = 0.0
        self.episode_action_reg_rew = 0.0

        return obs, {}

    def step(self, action):
        # unnormalize if action is in [-1, 1] range
        action = self.unnormalize_action(action)
        u = self.enforce_motor_limits(action)
        # x_dot: system dynamics
        x_dot = self.f(self.state, u).flatten()
        # euler update
        self.state = self.state + x_dot * self.dt
        # add floor boundary
        self.state[2] = max(self.state[2], 0)

        self.t += self.dt
        self.time_step += 1

        self.quadrotor.update_pose(self.state[0], self.state[1], self.state[2],
                                   self.state[3], self.state[4], self.state[5])
        if self.render_mode == 'human':
            self.render()

        # if within 0.1 meters of waypoint internally update to next waypoint
        target_waypoint = self.waypoints[self.target_waypoint_idx]
        distance_to_waypoint = np.linalg.norm(target_waypoint[:3] - self.state[:3])
        if (distance_to_waypoint < 1e-1) and (not self.visited_waypoints[self.target_waypoint_idx]):
            self.visited_waypoints[self.target_waypoint_idx] = True
            # move onto the next waypoint 
            self.target_waypoint_idx += 1
            reached_final_waypoint = self.target_waypoint_idx == len(self.waypoints)
            if reached_final_waypoint:
                # stay at final target waypoint until max time is reached
                self.target_waypoint_idx = len(self.waypoints) - 1

        x, y, z = self.state[0:3]
        terminated = self._is_out_of_bounds(x, y, z)
        truncated = self.time_step > self.max_time_steps - 1

        reward, rew_info = self._reward(self.state, u)
        
        # Accumulate reward components throughout the episode
        self.episode_position_rew += rew_info.get('position_rew', 0.0)
        self.episode_angle_rew += rew_info.get('angle_rew', 0.0)
        self.episode_velocity_rew += rew_info.get('velocity_rew', 0.0)
        self.episode_survival_rew += rew_info.get('survival_rew', 0.0)
        self.episode_action_reg_rew += rew_info.get('action_reg_rew', 0.0)
        
        # Prepare info dict - Monitor wrapper extracts these at episode end
        info = {}
        if terminated or truncated:
            # At episode end, provide cumulative totals
            # Monitor wrapper will extract these values (see monitor.py line 101-102)
            info['position_rew'] = self.episode_position_rew
            info['angle_rew'] = self.episode_angle_rew
            info['velocity_rew'] = self.episode_velocity_rew
            info['survival_rew'] = self.episode_survival_rew
            info['action_reg_rew'] = self.episode_action_reg_rew
        
        return self._get_obs(), reward, terminated, truncated, info

    def get_nearest_reference(self, current_position, nearby_timesteps=200):
        '''
        Get nearest reference point spatially.
        Looks nearby_timesteps * dt seconds ahead and behind to be lenient on the nearest reference point.
        Start of window will shrink as time runs out near the end.
        '''
        time_step = min(self.time_step, self.max_reference_time_steps - 1)
        stop_time_step = min(time_step + nearby_timesteps//2, self.max_reference_time_steps)
        start_time_step = max(stop_time_step - nearby_timesteps, 0)
        if self.time_step >= self.max_reference_time_steps - 1:
            # start reducing window size
            start_time_step = min(time_step - nearby_timesteps//2, self.max_reference_time_steps - 1)

        nearby_references = self.reference[:3, start_time_step : stop_time_step]
        diffs = nearby_references - current_position[:, None]
        dists = np.linalg.norm(diffs, axis=0)
    
        # nearest index must add on start time_step
        nearest_idx = np.argmin(dists) + start_time_step
        
        return self.reference[:, nearest_idx]


    def get_current_reference(self):
        '''
        Get reference point according to time step from smooth trajectory planning.
        '''
        time_step = min(self.time_step, self.max_reference_time_steps - 1)
        return self.reference[:, time_step]

    def _get_obs(self):
        current_reference = self.get_current_reference()
        if self.fully_observable:
            # return full state (all 12 variables) plus reference
            return np.concatenate([self.state, current_reference], dtype=np.float32)
        else:
            # return only noisy sensor measurements (y = [x, y, z, yaw]) plus reference
            obs = self.quadrotor.get_sensor_measurements()
            return np.concatenate([obs, current_reference], dtype=np.float32)
    
    def _is_out_of_bounds(self, x, y, z):
        out_of_bounds = z < 0 or abs(x) > self.boundary_length/2 or abs(y) > self.boundary_length/2 or z > self.boundary_length
        return out_of_bounds

    def render(self):
        # render every 0.1 seconds
        if self.time_step % 10 == 0:
            # try to stay real-time
            time_to_wait = self.start_time + self.t - time.time()
            while time_to_wait > 0:
                time.sleep(0.9 * time_to_wait)
                time_to_wait = self.t - time.time()
            # update x, y, z, roll, pitch, and yaw on GUI
            self.quadrotor.render_pose()

    def close(self):
        plt.ioff()
        plt.close()


if __name__ == '__main__':
    show_gui = True
    fully_observable = True
    # Use waypoints=None to trigger the new default randomized Lissajous path behavior
    waypoints = None
    num_waypoints = 64

    time_per_waypoint = 10.0 / num_waypoints

    reference_trajectory_time = num_waypoints * time_per_waypoint

    if show_gui:
        render_mode = 'human'
    else:
        render_mode = None
    
    add_takeoff_waypoint = False
    env = QuadrotorEnv(waypoints=waypoints, total_time=reference_trajectory_time, control_motors=True, normalized_actions=True,
                       render_mode=render_mode, fully_observable=fully_observable,
                       add_takeoff_waypoint=add_takeoff_waypoint)

    model = lambda obs: env.action_space.sample()
    if not show_gui:
        num_episodes = 100
        reward = 0
        for i in range(num_episodes):
            reward += run_single_episode(model, env)
        print("Avg Episode Reward: {}".format(reward / num_episodes))
    else:
        run_single_episode(model, env)