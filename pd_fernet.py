#!/usr/bin/python
#PID Trajectory tracking
# import time
import time as tm
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
# import scipy.linalg as la
import os
import copy
import matplotrecorder



show_animation = True
old_nearest_point_index = None

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 10.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.5 # robot radius [m]


# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0


try:
    from quintic_polynomials_planner import QuinticPolynomial
    import cubic_spline_planner
except ImportError:
    raise



class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp





class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0,omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.omega=omega
# 
def agent_state_update(state,v_in,omega_in, dt):
    state.x = state.x + v_in * math.cos(state.yaw) * dt
    state.y = state.y + v_in * math.sin(state.yaw) * dt
    state.yaw = pi_2_pi(state.yaw + omega_in*dt)
    state.v = v_in
    state.omega = omega_in
    return state

def PIDControl_track(Kp, error):
    error_tune=Kp*(error)
    return error_tune

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def calc_target_index(state, cx, cy):
    global old_nearest_point_index
    if old_nearest_point_index is None:
        dx = [state.x - icx for icx in cx]
        dy = [state.y - icy for icy in cy]
        d = [abs(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
        ind = d.index(min(d))
        old_nearest_point_index = ind
    else:
        ind = old_nearest_point_index
        distance_this_index = calc_distance(state, cx[ind], cy[ind])
        while True:
            ind = ind + 1 if (ind + 1) < len(cx) else ind
            distance_next_index = calc_distance(state, cx[ind], cy[ind])
            if distance_this_index < distance_next_index:
                break
            distance_this_index = distance_next_index
        old_nearest_point_index = ind
    L = 0.0
    k = 0.03  # look forward gain
    Lfc = 0.2  # look-ahead distance
    Lf = k * state.v + Lfc
    # search look ahead target point index
    while Lf > L and (ind + 1) < len(cx):
        dx = cx[ind] - state.x
        dy = cy[ind] - state.y
        L = math.sqrt(dx ** 2 + dy ** 2)
        ind += 1

    return ind 

def calc_distance(state, point_x, point_y):
    dx = state.x - point_x
    dy = state.y - point_y
    return math.sqrt(dx ** 2 + dy ** 2)

def calc_nearest_index(state, cx, cy, cyaw):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    mind = min(d)
    ind = d.index(mind)
    mind = math.sqrt(mind)
    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1
    return ind

def plot_arrow(x, y, yaw, length=1, width=2, fc="r", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
         plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def follwer_target(Lstate, Fstate, gap):
        x0 = Lstate.x
        y0 = Lstate.y

        x1 = Fstate.x
        y1 = Fstate.y

        dt = gap

        dist = np.sqrt( np.square(x0 - x1) + np.square(y0 - y1))
        # print(dist)
        
        t = dt/dist  #ratio
        print(t)

        x_des = (1 - t)*x0 + t*x1
        y_des = (1 - t)*y0 + t*y1

        return x_des, y_des, t

def  agent_angSpeed_Dpoint(state,des_x, des_y):
    yaw_new = (np.rad2deg(pi_2_pi(state.yaw)))
    
    if yaw_new < 0:
        yaw_new = yaw_new + 360
    dy = -state.y + des_y
    dx = -state.x + des_x

    theta = np.rad2deg(pi_2_pi(math.atan2(dy, dx)))
    if theta < 0:
        theta = theta + 360

    error = (theta - yaw_new)
    if error > 180:
        error = error - 360
    if error <- 180:
        error = error + 360
    #Kp_track tune wisely
    omega = 0.5*error

    return omega

def search_index(master_ind, master_state, cx, cy, gap):
    i,d = master_ind, 0
    while (d < (gap)):   #0.5 is small threshold for smooth working
        i = i - 1
        d = np.sqrt( np.square(master_state.x - cx[i]) + np.square(master_state.y - cy[i]))
    # print(d)
    return i

def dist_error(L_state, F_state, targDist):
    dist = np.sqrt( np.square(L_state.x - F_state.x) + np.square(L_state.y - F_state.y))
    error = (targDist - dist)
    return error


def agent_linSpeed(v_d, L_error, L_error_prev, F_error, F_error_prev, dt):
    v_in = v_d

    L_pd_val = pd_control(L_error, L_error_prev, 2, 0.01, dt)
    F_pd_val = pd_control(F_error, F_error_prev, 2, 0.01, dt)

    if np.absolute(L_error) > 0.05:
            v_in = v_d - L_pd_val
    if np.absolute(F_error) > 0.05:
            v_in = v_d + F_pd_val

    return v_in


def pd_control(error, prev_error, Kp, Kd, dt):
    pd_val = Kp* error + Kd*((error - prev_error)/dt)
    return pd_val

def main():
    #ax = [0,10,20,30,40,50,60,70,80]  ##still creating few problem
    # ax = [-15, -10.0, -5.0,  0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0 ]
    # ay = [0.0,   0.0,  0.0,  0.0, 3.0,  0.0, -3.0,  0.0,  0.0,  0.0]
    ax = [-20, -10, -5.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
    ay = [0.0, 0.0, 0.0, 0.0, 5.0,  0.0, -5.0,  0.0,  0.0,  0.0]
    #ay = [math.sin(ix / 5.0) * ix / 2.0 for ix in ax]
    goal = [ax[-1], ay[-1]]
    # cx, cy, cyaw, ck_curv, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    # ck = np.absolute(ck_curv) * [10]
    
    # lastIndex = len(cx) - 1
    T, time, t = 100, 0.0, [0]
    gap = 4

    state1 = State(x=-5, y=0, yaw=0, v=0.0,omega=0.0)
    state2 = State(x=-11, y=0, yaw=0, v=0.0,omega=0.0)
    state3 = State(x=-17, y=0, yaw=0, v=0.0,omega=0.0)
    # state4 = State(x=-15, y=0, yaw=0, v=0.0,omega=0.0)
    # target_ind1 = calc_target_index(state1, cx, cy)

    x1, y1, yaw1, v1, omega1 = [state1.x], [state1.y], [state1.yaw], [state1.v], [state1.omega]
    x2, y2, yaw2, v2, omega2 = [state2.x], [state2.y], [state2.yaw], [state2.v], [state2.omega]
    x3, y3, yaw3, v3, omega3 = [state3.x], [state3.y], [state3.yaw], [state3.v], [state3.omega]
    # x4, y4, yaw4, v4, omega4 = [state4.x], [state4.y], [state4.yaw], [state4.v], [state4.omega]

    # obstacle lists
    ob = np.array([[0, -3],
                    [10, 8],
                    [20.0, -2.0],
                   [25.0, -5.0],
                   [30.0, 0.0],
                   [40.0, 3.0],
                   [50.0, 5.0]
                   ])

    tx, ty, tyaw, tc, csp = generate_target_course(ax, ay)

    # initial state
    c_speed = 30 / 3.6  # current speed [m/s]
    
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 16.0  # current course position
    c_d = 0.0  # current lateral position [m]

    cox = [-15.0, -14.9, -14.8, -14.7, -14.6, -14.5, -14.4, -14.3, -14.2, -14.1, -14.0, -13.9, -13.8, -13.7, -13.6, -13.5, -13.4, -13.3, -13.2, -13.1, -13.0, -12.9, -12.8, -12.7, -12.6, -12.5, -12.4, -12.3, -12.2, -12.1, -12.0, -11.9, -11.8, -11.7, -11.6, -11.5, -11.4, -11.3, -11.2, -11.1, -11.0, -10.899999999999999, -10.8, -10.7, -10.6, -10.5, -10.399999999999999, -10.3, -10.2, -10.1, -10.0, -9.899999999999999, -9.8, -9.7, -9.6, -9.5, -9.399999999999999, -9.3, -9.2, -9.1, -9.0, -8.899999999999999, -8.8, -8.7, -8.6, -8.5, -8.399999999999999, -8.3, -8.2, -8.1, -8.0, -7.8999999999999995, -7.8, -7.699999999999999, -7.6, -7.5, -7.3999999999999995, -7.3, -7.199999999999999, -7.1, -7.0, -6.9, -6.799999999999999, -6.699999999999999, -6.6, -6.5, -6.4, -6.299999999999999, -6.199999999999999, -6.1, -6.0, -5.9, -5.799999999999999, -5.699999999999999, -5.6, -5.5, -5.399999999999999, -5.299999999999999, -5.199999999999999, -5.1, -5.0, -4.899999999999999, -4.799999999999999, -4.699999999999999, -4.6, -4.5, -4.399999999999999, -4.299999999999999, -4.199999999999999, -4.1, -4.0, -3.8999999999999986, -3.799999999999999, -3.6999999999999993, -3.5999999999999996, -3.5, -3.3999999999999986, -3.299999999999999, -3.1999999999999993, -3.0999999999999996, -3.0, -2.8999999999999986, -2.799999999999999, -2.6999999999999993, -2.5999999999999996, -2.5, -2.3999999999999986, -2.299999999999999, -2.1999999999999993, -2.0999999999999996, -2.0, -1.8999999999999986, -1.799999999999999, -1.6999999999999993, -1.5999999999999996, -1.5, -1.3999999999999986, -1.299999999999999, -1.1999999999999993, -1.0999999999999996, -1.0, -0.8999999999999986, -0.7999999999999989, -0.6999999999999993, -0.5999999999999996, -0.5, -0.3999999999999986, -0.29999999999999893, -0.1999999999999993, -0.09999999999999964]
    coy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    start_time = tm.time()
    last_time = start_time

    prev_err12 = dist_error(state1, state2, gap)
    prev_err23 = dist_error(state2, state3, gap)

    prev_AngErr1, prev_AngErr2, prev_AngErr3 = None, None, None

    while len(cox) < 191 :

        dt = 0.05
        v_ref = 120 / 3.6

        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]

        cox.append(path.x[1])
        coy.append(path.y[1])
        # target_ind1 = - 1
        # print(len(cox))


        #state1 omega calculate
        AngErr1 = agent_angSpeed_Dpoint(state1, path.x[1], path.y[1])

        if prev_AngErr1 == None:    prev_AngErr1 = AngErr1

        Kd = 0.005
        if np.absolute((AngErr1 - prev_AngErr1)) > 20:  Kd = 0.0

        omega1 = pd_control(AngErr1, prev_AngErr1, 0.75, Kd, dt)
        

        #state2 omega calculate
        des_x2, des_y2, t2 = follwer_target(state1, state2, gap)
        AngErr2 =  agent_angSpeed_Dpoint(state2, des_x2, des_y2)

        if prev_AngErr2 == None:    prev_AngErr2 = AngErr2

        Kd = 0.005
        if np.absolute((AngErr2 - prev_AngErr2)) > 20:  Kd = 0.0

        omega2 = pd_control(AngErr2, prev_AngErr2, 0.75, Kd, dt)
        

        #state3 omega calculate
        des_x3, des_y3, t3 = follwer_target(state2, state3, gap)
        AngErr3 =  agent_angSpeed_Dpoint(state3, des_x3, des_y3)

        if prev_AngErr3 == None:    prev_AngErr3 = AngErr3

        Kd = 0.005
        if np.absolute((AngErr3 - prev_AngErr3)) > 20:  Kd = 0.0
        omega3 = pd_control(AngErr3, prev_AngErr3, 0.75, Kd, dt)
        

        #agent linear speed
        err12 = dist_error(state1, state2, gap)
        err23 = dist_error(state2, state3, gap)

        # print(err12, err23)

        v_in1 = agent_linSpeed(v_ref, 0, 0, err12, prev_err12, dt)
        v_in2 = agent_linSpeed(v_ref, err12, prev_err12, err23, prev_err23, dt)
        v_in3 = agent_linSpeed(v_ref, err23, prev_err23, 0, 0, dt)

        #clipping of speed; if follower is too close then stop it
        if t2 > 1 : 
            v_in2 = 0.0
            omega2 = 0.0
        
        if t3 > 1 : 
            v_in3 = 0.0
            omega3 = 0.0

        #agent state update
        state1 = agent_state_update(state1, v_in1 , omega1, dt)
        state2 = agent_state_update(state2, v_in2, omega2, dt)
        state3 = agent_state_update(state3, v_in3, omega3, dt)



        prev_err12 = err12
        prev_err23 = err23
        prev_AngErr1 = AngErr1
        prev_AngErr2 = AngErr2
        prev_AngErr3 = AngErr3


        # last_time = current_time
        tm.sleep(.1)

        # time = time + dt
        x1.append(state1.x)
        y1.append(state1.y)
        x2.append(state2.x)
        y2.append(state2.y)
        x3.append(state3.x)
        y3.append(state3.y)
        # x4.append(state4.x)
        # y4.append(state4.y)


        if show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(path.x[1:], path.y[1:], "-or")
            # plot_circle(-5, -5, 1)
            # plt.plot(path.x[1:], path.y[1:],  

            # plt.plot(ob[:, 0], ob[:, 1], "xk")
            # plt.plot(ob[:, 0], ob[:, 1], marker = 'o', ms = 20, mec = 'w', mfc = 'b')
            plt.plot(ob[:, 0], ob[:, 1],  'o', ms=15, color='grey')


            # plt.plot(-1, -1, marker = 'o', ms = 20, mec = 'b', mfc = 'b')

            # plt.plot(des_x, des_y, "*")

            plot_arrow(state1.x, state1.y, state1.yaw, fc="m")
            plot_arrow(state2.x, state2.y, state2.yaw, fc="g")
            plot_arrow(state3.x, state3.y, state3.yaw, fc="b")
            # plot_arrow(state4.x, state4.y, state4.yaw, fc="c")

            # plt.plot(tx, ty)
            plt.plot(tx, ty, "-r", label="course")
            plt.plot(x1, y1, "-m", label="trajectory1")
            plt.plot(x2, y2, "-g", label="trajectory2")
            plt.plot(x3, y3, "-b", label="trajectory3")
            # plt.plot(x4, y4, "-c", label="trajectory4")

            # plt.plot(cox[target_ind1], coy[target_ind1], "*", label="target")
            # plt.plot(cox[target_ind2], coy[target_ind2], "*", label="target")
            # plt.plot(cx[target_ind3], cy[target_ind3], "*", label="target")
            # plt.plot(cx[target_ind4], cy[target_ind4], "*", label="target")

            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state1.v)[:4])
            plt.pause(0.01)
   
            # matplotrecorder.save_frame()    

    # matplotrecorder.save_movie("animation.gif", 0.1)


if __name__ == '__main__':
    main()

