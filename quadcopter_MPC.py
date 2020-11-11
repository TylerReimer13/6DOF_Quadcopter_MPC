import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import control
import cvxpy as cp
from math import sin as s, cos as c, atan2


def plot_results():
    plt.plot(x_pos, y_pos, 'b-', label='Quadcopter')
    plt.plot(spline_x_data, spline_y_data, 'r-', label='Trajectory')
    plt.plot(spline_gen.original_waypoints[:, 0], spline_gen.original_waypoints[:, 1], 'gx', label='Waypoints')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid()
    plt.legend()
    plt.title('Trajectory')
    plt.show()

    plt.plot(time, x_pos, label='x')
    plt.plot(time, y_pos, label='y')
    plt.plot(time, z_pos, label='z')
    plt.plot(time, x_vel, label='x vel')
    plt.plot(time, y_vel, label='y vel')
    plt.plot(time, z_vel, label='z vel')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()
    plt.title('Linear States')
    plt.show()

    plt.plot(time, roll, label='roll')
    plt.plot(time, pitch, label='pitch')
    plt.plot(time, yaw, label='yaw')
    plt.plot(time, roll_rate, label='roll rate')
    plt.plot(time, pitch_rate, label='pitch rate')
    plt.plot(time, yaw_rate, label='yaw rate')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()
    plt.title('Angular States')
    plt.show()

    plt.plot(time, ft_hist, label='ft')
    plt.plot(time, tx_hist, label='tx')
    plt.plot(time, ty_hist, label='ty')
    plt.plot(time, tz_hist, label='tz')
    plt.xlabel('Time (s)')
    plt.grid()
    plt.legend()
    plt.title('Control Inputs')
    plt.show()


def animated_plot2d():
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlim(-1., 6.)
    ax.set_ylim(-1., 6.)

    writer = animation.writers['ffmpeg']
    writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    def update(ii):
        quad_pos.set_data(x_pos[ii], y_pos[ii])
        spline.set_data(spline_x_data, spline_y_data)
        curr_goal.set_data(ref_x_hist[ii], ref_y_hist[ii])
        return (quad_pos,) + (spline,) + (curr_goal, )

    quad_pos, = plt.plot([], [], 'bX', markersize=6.)
    spline, = plt.plot([], [], 'r-', markersize=5.)
    curr_goal, = plt.plot([], [], 'gX', markersize=6.)
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    line_ani = animation.FuncAnimation(fig, update, sim_ctr, interval=5, repeat=True)
    # line_ani.save('/home/tyler/Videos/quadcopter_mpc_2d.mp4')
    plt.show()


def animated_plot3d():
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.set_xlim(-1., 6.)
    ax.set_ylim(-1., 6.)
    ax.set_zlim(0., 10.)

    writer = animation.writers['ffmpeg']
    writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    def update(ii):
        quad_pos.set_data(x_pos[ii], y_pos[ii])
        quad_pos.set_3d_properties(z_pos[ii])

        spline.set_data(spline_x_data, spline_y_data)
        spline.set_3d_properties(5.)
        return (quad_pos,) + (spline,)

    quad_pos, = plt.plot([], [], 'bX', markersize=10.)
    spline, = plt.plot([], [], 'r-', markersize=5.)
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    line_ani = animation.FuncAnimation(fig, update, sim_ctr, interval=5, repeat=True)
    # line_ani.save('/home/tyler/Videos/quadcopter_mpc_3d.mp4')
    plt.show()


class SplineGenerator:
    def __init__(self):
        self.waypoints = None
        self.spline_pts = None
        self.orig_waypoints = None

    @property
    def spline_data(self):
        return self.spline_pts.copy()

    @property
    def original_waypoints(self):
        return self.orig_waypoints

    def create_splines(self, waypts):
        self.waypoints = waypts.reshape((-1, 2))
        self.orig_waypoints = self.waypoints.copy()
        waypts = np.insert(waypts, 0, waypts[0], axis=0)
        waypts = np.insert(waypts, -1, waypts[-1], axis=0)
        splines = []
        for j in range(len(waypts)-3):
            this_spline = self.cubic_spline(waypts[j], waypts[j+1], waypts[j+2], waypts[j+3])
            splines.append(this_spline)

        self.spline_pts = np.array(splines).reshape((-1, 3))
        return self.spline_pts

    @staticmethod
    def cubic_spline(y0, y1, y2, y3, delt_mu=.001):
        mu = 0.
        points = []
        prev_x = 0.
        prev_y = 0.
        while mu <= 1.:
            mu2 = mu*mu
            a0 = y3 - y2 - y0 + y1
            a1 = y0 - y1 - a0
            a2 = y2 - y0
            a3 = y1
            mu += delt_mu
            point = a0*mu*mu2+a1*mu2+a2*mu+a3
            slope = atan2(point[1]-prev_y, point[0]-prev_x)
            point = np.append(point, slope)
            points.append(point)
            prev_x = point[0]
            prev_y = point[1]

        return points

    def plot(self):
        plt.plot(self.waypoints[:, 0], self.waypoints[:, 1], 'bx')
        plt.plot(self.spline_pts[:, 0], self.spline_pts[:, 1], 'r-')
        plt.show()


class Quadcopter:
    def __init__(self, **init_kwargs):
        self.Ix = 1.
        self.Iy = 1.
        self.Iz = 1.5

        self.g = 9.8  # m/s^2
        self.m = 5.

        #  States
        self.x_dot = init_kwargs['x_dot'] if 'x_dot' in init_kwargs.keys() else 0.
        self.y_dot = init_kwargs['y_dot'] if 'y_dot' in init_kwargs.keys() else 0.
        self.z_dot = init_kwargs['z_dot'] if 'z_dot' in init_kwargs.keys() else 0.
        self.x = init_kwargs['x'] if 'x' in init_kwargs.keys() else 0.
        self.y = init_kwargs['y'] if 'y' in init_kwargs.keys() else 0.
        self.z = init_kwargs['z'] if 'z' in init_kwargs.keys() else 0.

        self.roll_dot = init_kwargs['roll_dot'] if 'roll_dot' in init_kwargs.keys() else 0.
        self.pitch_dot = init_kwargs['pitch_dot'] if 'pitch_dot' in init_kwargs.keys() else 0.
        self.yaw_dot = init_kwargs['yaw_dot'] if 'yaw_dot' in init_kwargs.keys() else 0.
        self.roll = init_kwargs['roll'] if 'roll' in init_kwargs.keys() else 0.
        self.pitch = init_kwargs['pitch'] if 'pitch' in init_kwargs.keys() else 0.
        self.yaw = init_kwargs['yaw'] if 'yaw' in init_kwargs.keys() else 0.

        self.A_zoh = np.eye(12)
        self.B_zoh = np.zeros((12, 4))

        self.states = np.array([self.roll, self.pitch, self.yaw, self.roll_dot, self.pitch_dot, self.yaw_dot,
                                self.x_dot, self.y_dot, self.z_dot, self.x, self.y, self.z]).T

    @property
    def A(self):
        # Linear state transition matrix
        A = np.zeros((12, 12))
        A[0, 3] = 1.
        A[1, 4] = 1.
        A[2, 5] = 1.
        A[6, 1] = -self.g
        A[7, 0] = self.g
        A[9, 6] = 1.
        A[10, 7] = 1.
        A[11, 8] = 1.
        return A

    @property
    def B(self):
        # Control matrix
        B = np.zeros((12, 4))
        B[3, 1] = 1/self.Ix
        B[4, 2] = 1/self.Iy
        B[5, 3] = 1/self.Iz
        B[8, 0] = 1/self.m
        return B

    @property
    def C(self):
        C = np.eye(12)
        return C

    @property
    def D(self):
        D = np.zeros((12, 4))
        return D

    @property
    def Q(self):
        # State cost
        Q = np.eye(12)
        Q[8, 8] = 10.  # z vel
        Q[9, 9] = 10.  # x pos
        Q[10, 10] = 10.  # y pos
        Q[11, 11] = 100.  # z pos
        return Q

    @property
    def R(self):
        # Actuator cost
        R = np.eye(4)*.001
        return R

    def zoh(self):
        # Convert continuous time dynamics into discrete time
        sys = control.StateSpace(self.A, self.B, self.C, self.D)
        sys_discrete = control.c2d(sys, DT, method='zoh')

        self.A_zoh = np.array(sys_discrete.A)
        self.B_zoh = np.array(sys_discrete.B)

    def run_mpc(self, rx):
        cost = 0.
        constr = [x[:, 0] == x_init]
        for t in range(N):
            cost += cp.quad_form(rx - x[:, t], self.Q) + cp.quad_form(u[:, t], self.R)  # Linear Quadratic cost
            constr += [xmin <= x[:, t], x[:, t] <= xmax]  # State constraints
            constr += [x[:, t + 1] == self.A_zoh * x[:, t] + self.B_zoh * u[:, t]]

        cost += cp.quad_form(x[:, N] - rx, self.Q)  # End of trajectory error cost
        problem = cp.Problem(cp.Minimize(cost), constr)
        return problem

    def update_states(self, ft, tx, ty, tz):
        """
        Update the dynamic EOMs using Euler's forward method.
        EOMs taken from https://www.kth.se/polopoly_fs/1.588039.1600688317!/Thesis%20KTH%20-%20Francesco%20Sabatino.pdf
        """
        roll_ddot = ((self.Iy - self.Iz) / self.Ix) * (self.pitch_dot * self.yaw_dot) + tx / self.Ix
        pitch_ddot = ((self.Iz - self.Ix) / self.Iy) * (self.roll_dot * self.yaw_dot) + ty / self.Iy
        yaw_ddot = ((self.Ix - self.Iy) / self.Iz) * (self.roll_dot * self.pitch_dot) + tz / self.Iz
        x_ddot = -(ft/self.m) * (s(self.roll) * s(self.yaw) + c(self.roll)*c(self.yaw) * s(self.pitch))
        y_ddot = -(ft/self.m) * (c(self.roll) * s(self.yaw) * s(self.pitch) - c(self.yaw) * s(self.roll))
        z_ddot = -1*(self.g - (ft/self.m) * (c(self.roll) * c(self.pitch)))

        self.roll_dot += roll_ddot*DT
        self.roll += self.roll_dot*DT
        self.pitch_dot += pitch_ddot * DT
        self.pitch += self.pitch_dot * DT
        self.yaw_dot += yaw_ddot * DT
        self.yaw += self.yaw_dot * DT

        self.x_dot += x_ddot * DT
        self.x += self.x_dot * DT
        self.y_dot += y_ddot * DT
        self.y += self.y_dot * DT
        self.z_dot += z_ddot * DT
        self.z += self.z_dot * DT

        self.states = np.array([self.roll, self.pitch, self.yaw, self.roll_dot, self.pitch_dot, self.yaw_dot,
                                self.x_dot, self.y_dot, self.z_dot, self.x, self.y, self.z]).T

    def move_ref(self, curr_pt):
        curr_pos = np.array([self.x, self.y])
        if np.linalg.norm(curr_pos - curr_pt) <= waypt_thresh:
            return True
        return False

    def done(self, final_waypt):
        curr_pos = np.array([self.x, self.y])
        if np.linalg.norm(curr_pos - final_waypt) <= .15:
            return True
        return False

    def __call__(self, ft=0., tx=0., ty=0., tz=0.):
        hover = self.m*9.8
        self.update_states(hover+ft, tx, ty, tz)


if __name__ == "__main__":
    # Simulation and solver time step (lower is more accurate, but takes more time. Shouldn't go higher than .025)
    DT = .025

    # Defined desired waypoints to track
    waypoints = np.array([[0., 0.], [1., 2.], [2., 4.5], [3., 3.]])

    # Create intermediate trajectory points in between waypoints
    spline_gen = SplineGenerator()
    spline_data = spline_gen.create_splines(waypoints)

    spline_x_data = spline_gen.spline_data[:, 0]
    spline_y_data = spline_gen.spline_data[:, 1]
    spline_a_data = spline_gen.spline_data[:, 2]

    waypt_thresh = .25

    INF = np.inf
    xmin = np.array([-0.2, -0.2, -2*np.pi, -.25, -.25, -.25,  -INF,  -INF,  -INF, -INF, -INF, -INF])
    xmax = np.array([0.2,  0.2,   2*np.pi,  .25, .25,  .25,   INF,   INF,   INF,   INF,  INF, INF])

    # Initial quadcopter states
    init_dict = {'roll': 0., 'pitch': 0., 'yaw': 0., 'roll_dot': 0., 'pitch_dot': 0., 'yaw_dot': 0., 'x_dot': 0.,
                 'y_dot': 0., 'z_dot': 0., 'x': 0., 'y': 0., 'z': 5.}
    quad = Quadcopter(**init_dict)
    quad.zoh()

    # Initial solver states (copy of quadcopter states)
    x0 = np.array([init_dict['roll'], init_dict['pitch'], init_dict['yaw'], init_dict['roll_dot'],
                   init_dict['pitch_dot'], init_dict['yaw_dot'], init_dict['x_dot'], init_dict['y_dot'],
                   init_dict['x_dot'], init_dict['x'], init_dict['y'], init_dict['z']])

    # Desired states to track
    des_states = {'roll': 0., 'pitch': 0., 'yaw': 0., 'roll_dot': 0., 'pitch_dot': 0., 'yaw_dot': 0., 'x_dot': 0.,
                  'y_dot': 0., 'z_dot': 0., 'x': waypoints[0][0], 'y': waypoints[0][1], 'z': 5.}

    [nx, nu] = quad.B.shape
    N = 20  # MPC Horizon length

    # Convex optimization solver variables
    x = cp.Variable((nx, N+1))
    u = cp.Variable((nu, N))
    x_init = cp.Parameter(nx)

    nsim = 1000  # Number of simulation time steps

    x_pos = []
    y_pos = []
    z_pos = []
    x_vel = []
    y_vel = []
    z_vel = []
    roll = []
    pitch = []
    yaw = []
    roll_rate = []
    pitch_rate = []
    yaw_rate = []
    ft_hist = []
    tx_hist = []
    ty_hist = []
    tz_hist = []
    ref_x_hist = []
    ref_y_hist = []
    time = []

    idx_incr = 60  # Amount of 'lookahead' for trajectory follower
    idx = idx_incr  # Index for current spline point

    # Run simulation 'nsim' times
    sim_ctr = 0
    goal_found = False
    for i in range(1, nsim+1):
        ref_x = spline_x_data[idx]
        ref_y = spline_y_data[idx]
        ref_a = spline_a_data[idx]

        # If quadcopter gets close enough to current goal, move goal along the spline trajectory
        if quad.move_ref(np.array([ref_x, ref_y])):
            idx += idx_incr

        # If spline index tries to go past last waypoint
        if idx >= len(spline_x_data):
            idx = len(spline_x_data) - 1

        # Update reference states
        xr = np.array([des_states['roll'], des_states['pitch'], des_states['yaw'], des_states['roll_dot'],
                       des_states['pitch_dot'], des_states['yaw_dot'], des_states['x_dot'], des_states['y_dot'],
                       des_states['x_dot'], ref_x, ref_y, des_states['z']])

        # Run optimization for N horizons
        prob = quad.run_mpc(xr)

        # Solve convex optimization problem
        x_init.value = x0
        prob.solve(solver=cp.OSQP, warm_start=True)
        x0 = quad.A_zoh.dot(x0) + quad.B_zoh.dot(u[:, 0].value)

        # Send only first calculated command to quadcopter, then run optimization again
        quad(u[0, 0].value, u[1, 0].value, u[2, 0].value, u[3, 0].value)

        time.append(i*DT)
        x_pos.append(quad.x)
        y_pos.append(quad.y)
        z_pos.append(quad.z)
        x_vel.append(quad.x_dot)
        y_vel.append(quad.y_dot)
        z_vel.append(quad.z_dot)
        roll.append(quad.roll)
        pitch.append(quad.pitch)
        yaw.append(quad.yaw)
        roll_rate.append(quad.roll_dot)
        pitch_rate.append(quad.pitch_dot)
        yaw_rate.append(quad.yaw_dot)
        ft_hist.append(98.+u[0, 0].value)
        tx_hist.append(u[1, 0].value)
        ty_hist.append(u[2, 0].value)
        tz_hist.append(u[3, 0].value)
        ref_x_hist.append(ref_x)
        ref_y_hist.append(ref_y)

        sim_ctr += 1

        if quad.done(np.array([waypoints[-1][0], waypoints[-1][1]])):
            print('GOAL REACHED')
            goal_found = True
            plot_results()
            animated_plot2d()
            animated_plot3d()
            break

    if not goal_found:
        plot_results()
        animated_plot2d()
        animated_plot3d()


