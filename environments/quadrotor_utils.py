import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin
import sympy as sym

'''
Quadrotor dynamics and plotting code.
Dynamics adapted from Tim Bretl's course at the University of Illinois AE 353: Aerospace Control Systems
Plotting code adapted from Daniel Ingram, python robotics library.
'''
class Quadrotor():
    '''
    Quadrotor dynamics and plotting code
    '''
    def __init__(self, m, Jx, Jy, Jz, l, g=9.81, boundary_length=5, show_animation=True):
        '''
        m: mass of drone kg
        Jx, Jy, Jz: moments of intertia (kg·m^2)
        l: distance from center of drone to each rotor (m)
        g: gravity (m/s^2)
        boundary_length: side length of arena. assumed to be a cube
        show_animation: flag to show the quadrotor animation with matplotlib
        '''
        self.params = {
                        'm': m,
                        'Jx': Jx,
                        'Jy': Jy,
                        'Jz': Jz,
                        'l': l,
                        'g': g,
                      }
        self.boundary_length = boundary_length

        # rotor locations with last index as the homogeneous coord
        self.p1 = np.array([self.params['l'], 0, 0, 1]).T
        self.p2 = np.array([-self.params['l'], 0, 0, 1]).T
        self.p3 = np.array([0, self.params['l'], 0, 1]).T
        self.p4 = np.array([0, -self.params['l'], 0, 1]).T

        self.show_animation = show_animation

        self.pos_noise = 0.01
        self.yaw_noise = 0.001
        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            self.ax = fig.add_subplot(111, projection='3d')

        # get symbolic representation of system dynamics
        self.symbolic_state = self.get_symbolic_state_variables()
        self.symbolic_input = self.get_symbolic_input_variables()
        self.f = self.symbolic_system_dynamics()

    def get_symbolic_state_variables(self):
        # components of position (meters)
        p_x, p_y, p_z = sym.symbols('p_x, p_y, p_z')
        # yaw, pitch, roll angles (radians)
        psi, theta, phi = sym.symbols('psi, theta, phi')
        # components of linear velocity (meters / second)
        v_x, v_y, v_z = sym.symbols('v_x, v_y, v_z')
        # components of angular velocity (radians / second)
        w_x, w_y, w_z = sym.symbols('w_x, w_y, w_z')
        return (p_x, p_y, p_z), (psi, theta, phi), (v_x, v_y, v_z), (w_x, w_y, w_z)

    def get_symbolic_input_variables(self):
        # components of net rotor torque
        tau_x, tau_y, tau_z = sym.symbols('tau_x, tau_y, tau_z')
        # net rotor force
        f_z = sym.symbols('f_z')
        return tau_x, tau_y, tau_z, f_z

    def get_system_dynamics(self):
        '''
        Return a callable function for system dynamics

        Returns: f_num:
            f_num takes in a state and input
            x: state (12 dims) 
            u: input (4 dims)
        '''
        # unpack state insto symbolic matrix of size 12
        position, angle, velocity, angular_velocity = self.symbolic_state
        x = sym.Matrix([*position, *angle, *velocity, *angular_velocity])
        # components of net rotor torque and net vertical force as one matrix
        u = sym.Matrix(self.symbolic_input)  # [tau_x, tau_y, tau_z, f_z]
        # numerical representation of symbolic system
        f_num = sym.lambdify((x, u), self.f, modules='numpy')
        return f_num

    def symbolic_system_dynamics(self):
        '''
        Symbolic representation of system dynamics.
        '''
        position, angle, velocity, angular_velocity = self.symbolic_state
        # components of net rotor torque and net vertical force
        tau_x, tau_y, tau_z, f_z = self.symbolic_input
        # components of position (meters)
        p_x, p_y, p_z = position
        # euler angles (yaw, pitch, roll)
        psi, theta, phi = angle 
        # components of linear velocity (meters / second)
        v_x, v_y, v_z = velocity
        # components of angular velocity (radians / second)
        w_x, w_y, w_z = angular_velocity
        v_in_body = sym.Matrix([v_x, v_y, v_z])
        w_in_body = sym.Matrix([w_x, w_y, w_z])
        # parameters
        m = sym.nsimplify(self.params['m'])
        Jx = sym.nsimplify(self.params['Jx'])
        Jy = sym.nsimplify(self.params['Jy'])
        Jz = sym.nsimplify(self.params['Jz'])
        l = sym.nsimplify(self.params['l'])
        g = sym.nsimplify(self.params['g'])
        J = sym.diag(Jx, Jy, Jz)
        # rotation matrices
        Rz = sym.Matrix([[sym.cos(psi), -sym.sin(psi), 0], [sym.sin(psi), sym.cos(psi), 0], [0, 0, 1]])
        Ry = sym.Matrix([[sym.cos(theta), 0, sym.sin(theta)], [0, 1, 0], [-sym.sin(theta), 0, sym.cos(theta)]])
        Rx = sym.Matrix([[1, 0, 0], [0, sym.cos(phi), -sym.sin(phi)], [0, sym.sin(phi), sym.cos(phi)]])
        R_body_in_world = Rz @ Ry @ Rx
        # angular velocity to angular rates
        ex = sym.Matrix([[1], [0], [0]])
        ey = sym.Matrix([[0], [1], [0]])
        ez = sym.Matrix([[0], [0], [1]])
        M = sym.simplify(sym.Matrix.hstack((Ry @ Rx).T @ ez, Rx.T @ ey, ex).inv(), full=True)
        # applied forces
        f_in_body = R_body_in_world.T @ sym.Matrix([[0], [0], [-m * g]]) + sym.Matrix([[0], [0], [f_z]])
        # applied torques
        tau_in_body = sym.Matrix([[tau_x], [tau_y], [tau_z]])
        # equations of motion
        f = sym.Matrix.vstack(
            R_body_in_world * v_in_body,
            M * w_in_body,
            (1 / m) * (f_in_body - w_in_body.cross(m * v_in_body)),
            J.inv() * (tau_in_body - w_in_body.cross(J * w_in_body)),
        )
        # symbolic representation of equations of motion
        f = sym.simplify(f, full=True)
        return f
    
    def symbolic_sensor_model(self):
        position, angle, _, _ = self.get_symbolic_state_variables()
        p_x, p_y, p_z = position
        psi, _, _ = angle 
        g = sym.Matrix([p_x, p_y, p_z, psi])
        return g

    def get_sensor_measurements(self):
        # add noise to x, y, z
        position = np.array([self.x, self.y, self.z])
        x_hat = position + self.pos_noise * np.random.standard_normal(3)
        psi_hat = self.yaw + self.yaw_noise * np.random.standard_normal()

        measurement_max = np.array([self.boundary_length/2, self.boundary_length/2, self.boundary_length, np.pi])
        measurement_min= np.array([-self.boundary_length/2, -self.boundary_length/2, 0, -np.pi])

        measurements = np.hstack([x_hat, psi_hat]).clip(measurement_min, measurement_max)

        return measurements

    def initialize_flight_tracking(self, x=0, y=0, z=0, yaw=0, pitch=0, roll=0):
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.update_pose(x, y, z, yaw, pitch, roll)

    def update_pose(self, x, y, z, yaw, pitch, roll):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)

    def render_pose(self):
        if self.show_animation:
            self.plot()

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll), z]
             ])

    def set_waypoints(self, waypoints):
        self.waypoints= waypoints

    def set_reference_trajectory(self, reference):
        self.reference_trajectory = reference

    def plot(self):
        T = self.transformation_matrix()

        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        plt.cla()

        # plot waypoints and smooth reference trajectory in grey
        ref_pos = self.reference_trajectory[0:3]
        waypoints_xyz = self.waypoints[:, :3]
        self.ax.plot(ref_pos[0], ref_pos[1], ref_pos[2], linestyle=':', color='grey', label='Reference Pose')
        self.ax.scatter(waypoints_xyz[:, 0], waypoints_xyz[:, 1], waypoints_xyz[:, 2], color='grey', s=5)

        # plot drone 
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')

        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')

        # plot drone trajectory in black
        self.ax.plot(self.x_data, self.y_data, self.z_data, 'b:', label='Quadrotor Path')

        self.ax.set_xlim(-self.boundary_length / 2, self.boundary_length / 2)
        self.ax.set_ylim(-self.boundary_length / 2, self.boundary_length / 2)
        self.ax.set_zlim(0, self.boundary_length)   
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_zlabel('z (m)')

        plt.legend()
        plt.pause(0.0001)
        
class TrajectoryGenerator():
    """
    Generates a quintic polynomial trajectory.

    Author: Daniel Ingram (daniel-s-ingram)
    """
    def __init__(self, start_pos, des_pos, total_time,
                 start_vel=[0,0,0], des_vel=[0,0,0], start_acc=[0,0,0], des_acc=[0,0,0]):
        self.start_x = start_pos[0]
        self.start_y = start_pos[1]
        self.start_z = start_pos[2]

        self.des_x = des_pos[0]
        self.des_y = des_pos[1]
        self.des_z = des_pos[2]

        self.start_x_vel = start_vel[0]
        self.start_y_vel = start_vel[1]
        self.start_z_vel = start_vel[2]

        self.des_x_vel = des_vel[0]
        self.des_y_vel = des_vel[1]
        self.des_z_vel = des_vel[2]

        self.start_x_acc = start_acc[0]
        self.start_y_acc = start_acc[1]
        self.start_z_acc = start_acc[2]

        self.des_x_acc = des_acc[0]
        self.des_y_acc = des_acc[1]
        self.des_z_acc = des_acc[2]

        self.T = total_time

    def solve(self):
        A = np.array(
            [[0, 0, 0, 0, 0, 1],
             [self.T**5, self.T**4, self.T**3, self.T**2, self.T, 1],
             [0, 0, 0, 0, 1, 0],
             [5*self.T**4, 4*self.T**3, 3*self.T**2, 2*self.T, 1, 0],
             [0, 0, 0, 2, 0, 0],
             [20*self.T**3, 12*self.T**2, 6*self.T, 2, 0, 0]
            ])

        b_x = np.array(
            [[self.start_x],
             [self.des_x],
             [self.start_x_vel],
             [self.des_x_vel],
             [self.start_x_acc],
             [self.des_x_acc]
            ])

        b_y = np.array(
            [[self.start_y],
             [self.des_y],
             [self.start_y_vel],
             [self.des_y_vel],
             [self.start_y_acc],
             [self.des_y_acc]
            ])

        b_z = np.array(
            [[self.start_z],
             [self.des_z],
             [self.start_z_vel],
             [self.des_z_vel],
             [self.start_z_acc],
             [self.des_z_acc]
            ])

        self.x_c = np.linalg.solve(A, b_x)
        self.y_c = np.linalg.solve(A, b_y)
        self.z_c = np.linalg.solve(A, b_z)


    @staticmethod
    def calculate_position(c, t):
        """
        Calculates a position given a set of quintic coefficients and a time.

        Args
            c: List of coefficients generated by a quintic polynomial
                trajectory generator.
            t: Time at which to calculate the position

        Returns
            Position
        """
        return c[0] * t**5 + c[1] * t**4 + c[2] * t**3 + c[3] * t**2 + c[4] * t + c[5]

    @staticmethod
    def calculate_velocity(c, t):
        """
        Calculates a velocity given a set of quintic coefficients and a time.

        Args
            c: List of coefficients generated by a quintic polynomial
                trajectory generator.
            t: Time at which to calculate the velocity

        Returns
            Velocity
        """
        return 5 * c[0] * t**4 + 4 * c[1] * t**3 + 3 * c[2] * t**2 + 2 * c[3] * t + c[4]

    @staticmethod
    def calculate_acceleration(c, t):
        """
        Calculates an acceleration given a set of quintic coefficients and a time.

        Args
            c: List of coefficients generated by a quintic polynomial
                trajectory generator.
            t: Time at which to calculate the acceleration

        Returns
            Acceleration
        """
        return 20 * c[0] * t**3 + 12 * c[1] * t**2 + 6 * c[2] * t + 2 * c[3]

def run_single_episode(model, env, parameter=None):
    terminate = False
    truncate = False
    # Reset Environment
    obs, info = env.reset()
    rew = 0
    while not truncate and not terminate:
        if parameter is None:
            action = model(obs)
        else:
            action = model(obs, parameter)

        obs, reward, terminate, truncate, info = env.step(action)

        rew += reward
    return rew