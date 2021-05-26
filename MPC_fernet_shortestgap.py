#!/usr/bin/env python3
# coding: utf-8

import socket
import time

import matplotlib.pyplot as plt
# from scipy.optimize import minimize
import cvxpy
# import scipy
import math
import numpy as np
import cubic_spline_planner
import matplotrecorder
import copy



old_nearest_point_index = None
show_animation = True
NX = 3  # x = x, y, yaw
NU = 2  # a = [linear velocity,angular velocity ]
T = 5  # horizon length

# mpc parameters
R = np.diag([10000, 0.5])  # input cost matrix
Q = np.diag([10, 10, 0.000])  # state cost matrix
Qf = np.diag([10, 10, 0.000]) # state final matrix
Rd = np.diag([10000, 0.5])

GOAL_DIS = 4  # goal distance
STOP_SPEED = 0.15   # stop speed
MAX_TIME = 200.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

N_IND_SEARCH = 10  # Search index number
DT = 0.25  # [s] time tick

# TARGET_SPEED = 0.2  # [m/s] target speed

#---------------------------------------------------
#parameter
MAX_SPEED = 20.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 1.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 10.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5.0  # max prediction time [m]
MIN_T = 4.0  # min prediction time [m]
TARGET_SPEED = 10.0 / 3.6  # target speed [m/s]
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
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw

def get_linear_model_matrix(vref,phi):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[0, 2] = -vref*math.sin(phi)*DT
    A[1, 1] = 1.0
    A[1, 2] = vref*math.cos(phi)*DT
    A[2, 2] = 1.0

    B = np.zeros((NX, NU))
    B[0, 0] = DT * math.cos(phi)
    B[0, 1] = -0.5*DT*DT*math.sin(phi)*vref #0
    B[1, 0] = DT * math.sin(phi)
    B[1, 1] = 0.5*DT*DT*math.cos(phi)*vref #0
    B[2, 1] = DT

    return A, B


def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi
    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi
    return angle

def plot_arrow(x, y, yaw, length=0.2, width=0.5, fc="r", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def calc_ref_trajectory_Dpoint(state, des_x, des_y, v_ref, cur_vel, cur_omega):
    xref = np.zeros((NX, T + 1))
    vref = np.zeros((1, T + 1))
    # ncourse = len(cx)

    dx = -state.x + des_x
    dy = -state.y + des_y

    des_yaw = pi_2_pi(np.arctan2(dy, dx))

    xref[0, 0] = des_x
    xref[1, 0] = des_y
    xref[2, 0] = des_yaw 
    vref[0, 0] = v_ref
    travel = 0.0
    rotate = 0.0

    for i in range(T + 1):
        travel += abs(cur_vel) * DT
        rotate += cur_omega*DT
        
        xref[0, i] = des_x + travel*np.cos(state.yaw)
        xref[1, i] = des_y + travel*np.sin(state.yaw)
        xref[2, i] = des_yaw + rotate
        vref[0, i] = v_ref
            #print("if")

    return xref, vref


def update_state(state, v, omega):
    state.x = state.x + v * math.cos(state.yaw) * DT
    state.y = state.y + v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + omega * DT
    state.yaw = pi_2_pi(state.yaw)
    return state

def check_goal(state, goal, tind, nind,vi):
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)
    isgoal = (d <= GOAL_DIS)
    if abs(tind - nind) >= 5:
        isgoal = False
    isstop = (abs(vi) <= STOP_SPEED)
    if isgoal and isstop:
        return True
    return False

def iterative_linear_mpc_control(xref, x0, vref, ov, oomega):
    if ov is None or oomega is None:
        ov = [0.0] * T
        oomega = [0.0] * T
    for i in range(MAX_ITER):
        xbar = predict_motion(x0, ov, oomega, xref)
        pov, poomega = ov[:], oomega[:]
        ov, omega, ox, oy, oyaw = linear_mpc_control(xref, xbar, x0, vref)
        du = sum(abs(np.array(ov) - np.array(pov))) + sum(abs(np.array(oomega) - np.array(poomega)))
        if (du <= DU_TH):
            break   
    else:
        print("Iterative is max iter")
    return ov, omega, ox, oy, oyaw

def predict_motion(x0, ov, oomega, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0): #in-list #out-number,items
        xbar[i, 0] = x0[i]
    state = State(x=x0[0], y=x0[1], yaw=x0[2])
    for (i, v, omega) in zip(range(1, T + 1), ov, oomega):
        state = update_state(state, v, omega)
        xbar[0, i] = state.x
        xbar[1, i] = state.y 
        xbar[2, i] = state.yaw
    return xbar

def linear_mpc_control(xref, xbar, x0, vref):    
        
    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))
 
    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t] ,R)
        if t != 0:
            cost += cvxpy.quad_form(x[:, t], Q)        
        A, B = get_linear_model_matrix(vref[0,t],xbar[2,t])  
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]        
      
        if t < (T - 1):
            cost += cvxpy.quad_form((u[:, t + 1] - u[:, t]), Rd)
 
    cost += cvxpy.quad_form(x[:, T], Qf)
    constraints += [x[:, 0] == xref[:,0] - x0]    
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False,gp=False)
    #OSQP,CVXOPT, ECOS, scs

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        oyaw = get_nparray_from_matrix(x.value[2, :])
        ov = get_nparray_from_matrix(u.value[0, :])
        oomega = get_nparray_from_matrix(u.value[1, :])
        
        ox = ox + xref[0,:]
        oy = oy + xref[1,:]
        oyaw = oyaw + xref[2,:]
        ov = ov + vref[0,1:]
        oomega = -oomega

    else:
        print("Error: Cannot solve mpc..")
        ov, oomega, ox, oy, oyaw = None, None, None, None, None

    return ov, oomega, ox, oy, oyaw


def search_index(master_ind, master_state, cx, cy, gap):
    i,d = master_ind, 0
    while (d < (gap - 0.5)):   #0.5 is small threshold for smooth working
        i = i - 1
        d = np.sqrt( np.square(master_state.x - cx[i]) + np.square(master_state.y - cy[i]))
    # print(d)
    return i

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



if __name__ == '__main__':
    dl = 0.1 # course tick

    # ax = [-3,-2,-1,0,  2,4,6,8,10,12,14,16]
    # ay = [0, 0, 0, 0,  1,0,-1,0,0,0,0,0 ]
    ax = [-20, -10, -5.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0 ]
    ay = [0.0, 0.0, 0.0, 0.0, 5.0,  0.0, -5.0,  0.0,  0.0,  0.0]
    
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    # sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    # goal = [cx[-1], cy[-1]]
    # cyaw = smooth_yaw(cyaw)  

    state1 = State(x=-5, y=0, yaw=0)
    state2 = State(x=-11, y=0, yaw=0)
    state3 = State(x=-17, y=0, yaw=0)


    x1, y1, yaw1 = [state1.x], [state1.y], [state1.yaw]
    x2, y2, yaw2 = [state2.x], [state2.y], [state2.yaw]
    x3, y3, yaw3 = [state3.x], [state3.y], [state3.yaw]
    
    oomega1, ov1 = None, None
    oomega2, ov2 = None, None
    oomega3, ov3 = None, None

    vi1 = 0
    vi2 = 0
    vi3 = 0

    omegai3 = 0
    omegai2 = 0
    omegai1 = 0

    gap = 4

    #------------------------------------
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
    c_speed = TARGET_SPEED  # current speed [m/s]   
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 16.0  # current course position
    c_d = 0.0  # current lateral position [m]
    cox = [-15.0, -14.9, -14.8, -14.7, -14.6, -14.5, -14.4, -14.3, -14.2, -14.1, -14.0, -13.9, -13.8, -13.7, -13.6, -13.5, -13.4, -13.3, -13.2, -13.1, -13.0, -12.9, -12.8, -12.7, -12.6, -12.5, -12.4, -12.3, -12.2, -12.1, -12.0, -11.9, -11.8, -11.7, -11.6, -11.5, -11.4, -11.3, -11.2, -11.1, -11.0, -10.899999999999999, -10.8, -10.7, -10.6, -10.5, -10.399999999999999, -10.3, -10.2, -10.1, -10.0, -9.899999999999999, -9.8, -9.7, -9.6, -9.5, -9.399999999999999, -9.3, -9.2, -9.1, -9.0, -8.899999999999999, -8.8, -8.7, -8.6, -8.5, -8.399999999999999, -8.3, -8.2, -8.1, -8.0, -7.8999999999999995, -7.8, -7.699999999999999, -7.6, -7.5, -7.3999999999999995, -7.3, -7.199999999999999, -7.1, -7.0, -6.9, -6.799999999999999, -6.699999999999999, -6.6, -6.5, -6.4, -6.299999999999999, -6.199999999999999, -6.1, -6.0, -5.9, -5.799999999999999, -5.699999999999999, -5.6, -5.5, -5.399999999999999, -5.299999999999999, -5.199999999999999, -5.1, -5.0, -4.899999999999999, -4.799999999999999, -4.699999999999999, -4.6, -4.5, -4.399999999999999, -4.299999999999999, -4.199999999999999, -4.1, -4.0, -3.8999999999999986, -3.799999999999999, -3.6999999999999993, -3.5999999999999996, -3.5, -3.3999999999999986, -3.299999999999999, -3.1999999999999993, -3.0999999999999996, -3.0, -2.8999999999999986, -2.799999999999999, -2.6999999999999993, -2.5999999999999996, -2.5, -2.3999999999999986, -2.299999999999999, -2.1999999999999993, -2.0999999999999996, -2.0, -1.8999999999999986, -1.799999999999999, -1.6999999999999993, -1.5999999999999996, -1.5, -1.3999999999999986, -1.299999999999999, -1.1999999999999993, -1.0999999999999996, -1.0, -0.8999999999999986, -0.7999999999999989, -0.6999999999999993, -0.5999999999999996, -0.5, -0.3999999999999986, -0.29999999999999893, -0.1999999999999993, -0.09999999999999964]
    coy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



while True:

        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]


        #Robot 1 
        xref1, vref1 = calc_ref_trajectory_Dpoint(state1, path.x[1], path.y[1], TARGET_SPEED, vi1, omegai1)
        x0_1 = [state1.x, state1.y, state1.yaw] 
        ov1, oomega1, _, _, _ = iterative_linear_mpc_control(xref1, x0_1, vref1, ov1, oomega1) #solve MPC
        if oomega1 is not None:
            vi1 , omegai1 = ov1[0], oomega1[0]
        

        #Robot2
        des_x2, des_y2, t2 = follwer_target(state1, state2, gap)
        xref2, vref2 = calc_ref_trajectory_Dpoint(state2, des_x2, des_y2, TARGET_SPEED, vi2, omegai2)

        x0_2 = [state2.x, state2.y, state2.yaw] 
        ov2, oomega2, _, _, _ = iterative_linear_mpc_control(xref2, x0_2, vref2, ov2, oomega2)
        if oomega2 is not None:
            vi2 , omegai2 = ov2[0], oomega2[0]    
       

       #Robot3 
        des_x3, des_y3, t3 = follwer_target(state2, state3, gap)
        xref3, vref3 = calc_ref_trajectory_Dpoint(state3, des_x3, des_y3, TARGET_SPEED, vi3, omegai3)

        x0_3 = [state3.x, state3.y, state3.yaw] 
        ov3, oomega3, _, _, _ = iterative_linear_mpc_control(xref3, x0_3, vref3, ov3, oomega3)
        if oomega3 is not None:
            vi3 , omegai3 = ov3[0], oomega3[0]    

        #add clipping
        dist12 = np.sqrt( np.square(state1.x - state2.x) + np.square(state1.y - state2.y))
        dist23 = np.sqrt( np.square(state2.x - state3.x) + np.square(state2.y - state3.y))

        # if dist12 > 1.2:
        #     vi1 = 0   #stop leader
        #     print(dist12)

        # if dist23 > 1.2:
        #     vi2 = 0
        #     print(dist23)


        state1 = update_state(state1, vi1, omegai1)
        state2 = update_state(state2, vi2, omegai2)
        state3 =update_state(state3, vi3, omegai3)

        time.sleep(0.3)

        x1.append(state1.x)
        y1.append(state1.y)
        x2.append(state2.x)
        y2.append(state2.y)
        x3.append(state3.x)
        y3.append(state3.y)

        # print(target_ind1, target_ind2, target_ind3)

        if show_animation:  # pragma: no cover
            plt.cla()
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(ob[:, 0], ob[:, 1],  'o', ms=15, color='grey')



            plt.plot(cx, cy, "-r", label="course")

            plot_arrow(state1.x, state1.y, state1.yaw, fc="m")
            plot_arrow(state2.x, state2.y, state2.yaw, fc="g")
            plot_arrow(state3.x, state3.y, state3.yaw, fc="b")
            # plot_arrow(state4.x, state4.y, state4.yaw, fc="c")

            # plt.plot(tx, ty)
            # plt.plot(tx, ty, "-r", label="course")
            plt.plot(x1, y1, "-m", label="trajectory1")
            plt.plot(x2, y2, "-g", label="trajectory2")
            plt.plot(x3, y3, "-b", label="trajectory3")
            # plt.plot(x4, y4, "-c", label="trajectory4")

            plt.plot(path.x[1], path.y[1], "*", label="target")
            plt.plot(des_x2, des_y2, "*", label="target")
            plt.plot(des_x3, des_y3, "*", label="target")
            # plt.plot(cx[target_ind4], cy[target_ind4], "*", label="target")

            plt.axis("equal")
            plt.grid(True)
            # plt.title("Speed[km/h]:" + str(state1.v)[:4])
            plt.pause(0.01)

            # matplotrecorder.save_frame()

# matplotrecorder.save_movie("animation.gif", 0.1)
