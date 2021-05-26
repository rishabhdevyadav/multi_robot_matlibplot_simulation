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
R = np.diag([10000, 0.3])  # input cost matrix
Q = np.diag([10, 10, 0.000])  # state cost matrix
Qf = np.diag([10, 10, 0.000]) # state final matrix
Rd = np.diag([10000, 0.2])

GOAL_DIS = 4  # goal distance
STOP_SPEED = 0.15   # stop speed
MAX_TIME = 200.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

N_IND_SEARCH = 10  # Search index number
DT = 0.25  # [s] time tick

TARGET_SPEED = 0.2  # [m/s] target speed





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

def update_state_gazebo(state, x, y, yaw):
    state.x = float(x)
    state.y = float(y)
    state.yaw = float(yaw)
    state.yaw = pi_2_pi(state.yaw)
    return state

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


def calc_nearest_index_follower(state, cx, cy):
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

def calc_nearest_index(state, cx, cy, cyaw, pind):
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
    mind = min(d)
    ind = d.index(mind) + pind
    mind = math.sqrt(mind)
    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y
    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1
    return ind, mind

def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    # direction = 1.0  # forward
    # # Set stop point
    # for i in range(len(cx) - 1):
    #     dx = cx[i + 1] - cx[i]
    #     dy = cy[i + 1] - cy[i]
    #     move_direction = math.atan2(dy, dx)
    #     if dx != 0.0 and dy != 0.0:
    #         dangle = abs(pi_2_pi(move_direction - cyaw[i]))
    #         if dangle >= math.pi / 4.0:
    #             direction = -1.0
    #         else:
    #             direction = 1.0
    #     if direction != 1.0:
    #         speed_profile[i] = - target_speed
    #     else:
    #         speed_profile[i] = target_speed
    # speed_profile[-1] = 0.0
    return speed_profile


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind, cur_vel):
    xref = np.zeros((NX, T + 1))
    vref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind
    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = cyaw[ind]
    vref[0, 0] = sp[ind]
    travel = 0.0
    for i in range(T + 1):
        travel += abs(cur_vel) * DT
        dind = int(round(travel / dl))
        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = cyaw[ind + dind]
            vref[0, i] = sp[ind + dind]
            #print("if")
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = cyaw[ncourse - 1]
            vref[0, i] = sp[ncourse - 1]
            #print("else")
    return xref, ind, vref


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



def calc_ref_trajectory_follower(Lead_ind, Lead_state, cx, cy, cyaw, ck, sp, dl, pind,cur_vel):
    xref = np.zeros((NX, T + 1))
    vref = np.zeros((1, T + 1))
    ncourse = len(cx)
    
    gap = 1

    ind = search_index(Lead_ind, Lead_state, cx, cy, gap)

    # if pind >= ind:
    #     ind = pind
    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = cyaw[ind]
    vref[0, 0] = sp[ind]
    travel = 0.0
    for i in range(T + 1):
        travel += abs(cur_vel) * DT
        dind = int(round(travel / dl))
        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = cyaw[ind + dind]
            vref[0, i] = sp[ind + dind]
            #print("if")
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = cyaw[ncourse - 1]
            vref[0, i] = sp[ncourse - 1]
            #print("else")
    return xref, ind, vref


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


if __name__ == '__main__':
    dl = 0.1 # course tick

    ax = [-3,-2,-1,0,  2,4,6,8,10,12,14,16]
    ay = [0, 0, 0, 0,  1,0,-1,0,0,0,0,0 ]
    
    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    goal = [cx[-1], cy[-1]]
    # cyaw = smooth_yaw(cyaw)  

    state1 = State(x=0, y=0, yaw=0)
    state2 = State(x=-1, y=0, yaw=0)
    state3 = State(x=-2, y=0, yaw=0)


    x1, y1, yaw1 = [state1.x], [state1.y], [state1.yaw]
    x2, y2, yaw2 = [state2.x], [state2.y], [state2.yaw]
    x3, y3, yaw3 = [state3.x], [state3.y], [state3.yaw]

    target_ind1, _  = calc_nearest_index(state1, cx, cy, cyaw, 0)
    target_ind2, _  = calc_nearest_index(state2, cx, cy, cyaw, 0)
    target_ind3, _  = calc_nearest_index(state3, cx, cy, cyaw, 0)
    
    oomega1, ov1 = None, None
    oomega2, ov2 = None, None
    oomega3, ov3 = None, None

    vi1 = 0
    vi2 = 0
    vi3 = 0

    omegai3 = 0


while True:

        #Robot 1 (Leader Robot) is simple trajectory tracking
        xref1, target_ind1, vref1 = calc_ref_trajectory(state1, cx, cy, cyaw, ck, sp, dl, target_ind1, vi1)
        # x0_1 = [float(pos[0]), float(pos[1]), float(pos[2])]   #current Position
        x0_1 = [state1.x, state1.y, state1.yaw] 
        ov1, oomega1, _, _, _ = iterative_linear_mpc_control(xref1, x0_1, vref1, ov1, oomega1) #solve MPC
        if oomega1 is not None:
            vi1 , omegai1 = ov1[0], oomega1[0]
        # state1 = update_state_gazebo(state1, pos[0], pos[1], pos[2]) 
        

        #Robot2
        #it using "search_index()" to calculate target index for itself
        #search index takes Leader current state 
        nearest_ind2 = calc_nearest_index_follower(state2, cx, cy)
        xref2, target_ind2, vref2 = calc_ref_trajectory_follower(target_ind1, state1, cx, cy, cyaw, ck, sp, dl, target_ind2, vi2) 
        # x0_2 = [float(pos[3]), float(pos[4]), float(pos[5])]    
        x0_2 = [state2.x, state2.y, state2.yaw] 
        ov2, oomega2, _, _, _ = iterative_linear_mpc_control(xref2, x0_2, vref2, ov2, oomega2)
        if oomega2 is not None:
            vi2 , omegai2 = ov2[0], oomega2[0]    
            if target_ind2 - nearest_ind2 < 2:  #add clipping
                vi2 = 0                         #stop follower
                omegai2 = 0
        # state2 = update_state_gazebo(state2, pos[3], pos[4], pos[5])

        

        nearest_ind3 = calc_nearest_index_follower(state3, cx, cy)
        xref3, target_ind3, vref3 = calc_ref_trajectory_follower(target_ind2, state2, cx, cy, cyaw, ck, sp, dl, target_ind3, vi3) 

        # x0_3 = [float(pos[6]), float(pos[7]), float(pos[8])]    
        x0_3 = [state3.x, state3.y, state3.yaw] 
        ov3, oomega3, _, _, _ = iterative_linear_mpc_control(xref3, x0_3, vref3, ov3, oomega3)
        if oomega3 is not None:
            vi3 , omegai3 = ov3[0], oomega3[0]    
            if target_ind3 - nearest_ind3 < 2: 
                vi3 = 0
                omegai3 = 0
        # state3 = update_state_gazebo(state3, pos[6], pos[7], pos[8])


        #add clipping
        dist12 = np.sqrt( np.square(state1.x - state2.x) + np.square(state1.y - state2.y))
        dist23 = np.sqrt( np.square(state2.x - state3.x) + np.square(state2.y - state3.y))

        if dist12 > 1.2:
            vi1 = 0   #stop leader
            print(dist12)

        if dist23 > 1.2:
            vi2 = 0
            print(dist23)


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

        print(target_ind1, target_ind2, target_ind3)

        if show_animation:  # pragma: no cover
            plt.cla()

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

            # plt.plot(cox[target_ind1], coy[target_ind1], "*", label="target")
            # plt.plot(cox[target_ind2], coy[target_ind2], "*", label="target")
            # plt.plot(cx[target_ind3], cy[target_ind3], "*", label="target")
            # plt.plot(cx[target_ind4], cy[target_ind4], "*", label="target")

            plt.axis("equal")
            plt.grid(True)
            # plt.title("Speed[km/h]:" + str(state1.v)[:4])
            plt.pause(0.01)
   
            # matplotrecorder.save_frame()    

    # matplotrecorder.save_movie("animation.gif", 0.1)
