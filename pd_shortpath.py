#!/usr/bin/python
#PID Trajectory tracking
import time as tm
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

import copy
import matplotrecorder

# dt = 0.06
N_IND_SEARCH = 8

Kp_track=0.5
show_animation = True
old_nearest_point_index = None

import cubic_spline_planner
# import time

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


def calc_nearest_index_leader(state, cx, cy, cyaw, pind):
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
    return ind+1, mind


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

def plot_arrow(x, y, yaw, length=.2, width=0.4, fc="r", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
         plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def agent_angErr(state,cx,cy,target_ind):
    yaw_new = (np.rad2deg(pi_2_pi(state.yaw)))
    if yaw_new < 0:
        yaw_new = yaw_new + 360
    dy = -state.y + cy[target_ind] 
    dx = -state.x + cx[target_ind]

    theta = np.rad2deg(pi_2_pi(math.atan2(dy, dx)))
    if theta < 0:
        theta = theta + 360

    error = (theta - yaw_new)
    if error > 180:
        error = error - 360
    if error <- 180:
        error = error + 360

    return error*0.01

def search_index(master_ind, master_state, cx, cy, gap):
    i,d = master_ind, 0
    while (d < (gap)):   #0.5 is small threshold for smooth working
        i = i - 1
        d = np.sqrt( np.square(master_state.x - cx[i]) + np.square(master_state.y - cy[i]))
    # print(d)
    return i

def search_index_via_path(master_ind, master_state, cx, cy, gap):
    i = master_ind
    d = np.sqrt( np.square(master_state.x - cx[i]) + np.square(master_state.y - cy[i]))
    d = 0
    while (d < (gap)):
        # i = i - 1
        d = d + np.sqrt( np.square(cx[i] - cx[i-1]) + np.square(cy[i] - cy[i-1]))
        i = i - 1
    return i

def dist_error(L_state, F_state, targDist):
    dist = np.sqrt( np.square(L_state.x - F_state.x) + np.square(L_state.y - F_state.y))
    error = (-targDist + dist)
    return error


def agent_linSpeed(v_d, L_error, F_error):
    v_in = v_d
    if np.absolute(L_error) > 0:
            v_in = v_d - (0.005*L_error)
    if np.absolute(F_error) > 0:
            v_in = v_d + (0.005*F_error)

    return v_in


def pd_control(error, prev_error, Kp, Kd, dt):
    # errInt = errInt +  error*dt
    pd_val = Kp* error + Kd*((error - prev_error)/dt)

    return pd_val




def main():
    #ax = [0,10,20,30,40,50,60,70,80]  ##still creating few problem
    ax = [-4,-3, -2,-1,0,  2,4,6,8,10,12,14,16]
    ay = [0,0,  0, 0, 0,  1,0,-1,0,0,0,0,0 ]
    
    alpha = 10
    t = np.linspace(0, 2*np.pi, num=100)

    ax = alpha * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1)
    ay = alpha * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

    t = 0.16
    ax1 = alpha * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1)
    ay1 = alpha * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)


    t = 0.08
    ax2 = alpha * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1)
    ay2 = alpha * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

    t = 0.0
    ax3 = alpha * np.sqrt(2) * np.cos(t) / (np.sin(t)**2 + 1)
    ay3 = alpha * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

    print(ax1, ay1)
    print(ax2, ay2)
    print(ax3, ay3)
    print("---------")

    goal = [ax[-1], ay[-1]]
    cx, cy, cyaw, ck_curv, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    ck = np.absolute(ck_curv) * [10]
    
    T, time, t = 100, 0.0, [0]
    gap = 1

    state1 = State(x=ax1, y=ay1, yaw=np.deg2rad(120), v=0.0,omega=0.0)
    state2 = State(x=ax2, y=ay2, yaw=np.deg2rad(105), v=0.0,omega=0.0)
    state3 = State(x=ax3, y=ay3, yaw=np.deg2rad(95), v=0.0,omega=0.0)

    x1, y1, yaw1, v1, omega1 = [state1.x], [state1.y], [state1.yaw], [state1.v], [state1.omega]
    x2, y2, yaw2, v2, omega2 = [state2.x], [state2.y], [state2.yaw], [state2.v], [state2.omega]
    x3, y3, yaw3, v3, omega3 = [state3.x], [state3.y], [state3.yaw], [state3.v], [state3.omega]
    # x4, y4, yaw4, v4, omega4 = [state4.x], [state4.y], [state4.yaw], [state4.v], [state4.omega]

    v_ref = 0.3

    start_time = tm.time()
    last_time = start_time

    # if using "calc_target_index" here then use it in while loop also
    tar_ind1 = calc_target_index(state1, cx, cy)
    cx_short = []
    cy_short = []
    cyaw_short = []

    # print(near_ind1+20, near_ind3+20)
    for i in range(-30, tar_ind1+20):
        cx_short.append(cx[i])
        cy_short.append(cy[i])
        cyaw_short.append(cyaw[i])

    i = 0


    target_ind1 = calc_nearest_index(state1, cx_short, cy_short, cyaw_short) + 3
    target_ind2 = search_index_via_path(target_ind1, state1, cx_short, cy_short, gap)
    target_ind3 = search_index_via_path(target_ind2, state2, cx_short, cy_short, gap)

    AngErr1 = agent_angErr(state1,cx_short,cy_short,target_ind1)
    AngErr2 = agent_angErr(state2,cx_short,cy_short,target_ind2)
    AngErr3 = agent_angErr(state3,cx_short,cy_short,target_ind3)

    print(AngErr1/0.01, AngErr2/0.01, AngErr3/0.01)

    prev_AngErr1, prev_AngErr2, prev_AngErr3 = AngErr1, AngErr2, AngErr3

    PosErr1 = 0
    PosErr2 = dist_error(state1, state2, gap)
    PosErr3 = dist_error(state2, state3, gap)
    prev_PosErr1, prev_PosErr2, prev_PosErr3 = PosErr1, PosErr2, PosErr3

    print(PosErr2, PosErr3)



    while True:
        
        
        if tar_ind1 < np.size(cx)-1:
            tar_ind1 = calc_target_index(state1, cx, cy)
            # tar_ind1 = calc_nearest_index(state1, cx, cy, cyaw)


            if np.size(cx) > tar_ind1 + 20:
        
                cx_short.append(cx_short.pop(0)) 
                cx_short[-1] = cx[tar_ind1 + 20]

                cy_short.append(cy_short.pop(0)) 
                cy_short[-1] = cy[tar_ind1 + 20]

                cyaw_short.append(cyaw_short.pop(0)) 
                cyaw_short[-1] = cyaw[tar_ind1 + 20]

            else:
                cx_short.append(cx_short.pop(0)) 
                cx_short[-1] = cx[tar_ind1 + 20 - np.size(cx)]

                cy_short.append(cy_short.pop(0)) 
                cy_short[-1] = cy[tar_ind1 + 20- np.size(cy)]

                cyaw_short.append(cyaw_short.pop(0)) 
                cyaw_short[-1] = cyaw[tar_ind1 + 20 - np.size(cyaw)]

                
        else:
                cx_short.append(cx_short.pop(0)) 
                cx_short[-1] = cx[tar_ind1 + 20 - np.size(cx) + i]

                cy_short.append(cy_short.pop(0)) 
                cy_short[-1] = cy[tar_ind1 + 20- np.size(cy) + i]

                cyaw_short.append(cyaw_short.pop(0)) 
                cyaw_short[-1] = cyaw[tar_ind1 + 20 - np.size(cyaw) +i]
                i = i + 1


        current_time = tm.time()
        dt = current_time - last_time
        dt = 0.3

        # target_ind1 = calc_target_index(state1, cx_short, cy_short) 
        target_ind1 = calc_nearest_index(state1, cx_short, cy_short, cyaw_short) + 5
        AngErr1 = agent_angErr(state1,cx_short,cy_short,target_ind1)
        omega1 = pd_control(AngErr1, prev_AngErr1, 3, 0.1, dt = 0.1)
        


        target_ind2 = search_index_via_path(target_ind1, state1, cx_short, cy_short, gap)
        nearest_ind2 = calc_nearest_index(state2, cx_short, cy_short, cyaw_short)

        AngErr2 = agent_angErr(state2,cx_short,cy_short,target_ind2)
        omega2 = pd_control(AngErr2, prev_AngErr2, 3, 0.1, dt = 0.1)


        target_ind3 = search_index_via_path(target_ind2, state2, cx_short, cy_short, gap)
        nearest_ind3 = calc_nearest_index(state3, cx_short, cy_short, cyaw_short)

        AngErr3 = agent_angErr(state3,cx_short,cy_short,target_ind3)
        omega3 = pd_control(AngErr3, prev_AngErr3, 3, 0.1, dt = 0.1)


        err12 = dist_error(state1, state2, gap)
        err23 = dist_error(state2, state3, gap)

        PosErr1 = 0.0
        PosErr2 = err12
        PosErr3 = err23
        v_in1 = pd_control(PosErr1, prev_PosErr1, 1, 0.2, dt = 0.1) + v_ref
        v_in2 = pd_control(PosErr2, prev_PosErr2, 1, 0.2, dt = 0.1) + v_ref
        v_in3 = pd_control(PosErr3, prev_PosErr3, 1, 0.2, dt = 0.1) + v_ref

        # v_in1 = agent_linSpeed(v_ref, L_error=0, F_error=err12)
        # v_in2 = agent_linSpeed(v_ref, L_error=err12, F_error=err23)
        # v_in3 = agent_linSpeed(v_ref, L_error=err23, F_error=0)
        
        if target_ind2 - nearest_ind2 < 5:
            v_in2 = 0
            omega2 = 0
        
        if target_ind3 - nearest_ind3 < 5:
            v_in3 = 0
            omega3 = 0

        state1 = agent_state_update(state1, v_in1, omega1, dt)
        state2 = agent_state_update(state2, v_in2, omega2, dt)
        state3 = agent_state_update(state3, v_in3, omega3, dt)

        # print(target_ind1, target_ind2, target_ind3)

        time = time + dt
        x1.append(state1.x)
        y1.append(state1.y)
        x2.append(state2.x)
        y2.append(state2.y)
        x3.append(state3.x)
        y3.append(state3.y)

        t.append(time)

        last_time = current_time
        prev_AngErr1 = AngErr1
        prev_AngErr2 = AngErr2
        prev_AngErr3 = AngErr3

        prev_PosErr1 = PosErr1
        prev_PosErr2 = PosErr2
        prev_PosErr3 = PosErr3


        tm.sleep(0.25)

        # print(state1.x, state1.y, state1.yaw)
        # print(state2.x, state2.y, state2.yaw)
        # print(state3.x, state3.y, state3.yaw)
        print("-----------")

        if show_animation:  # pragma: no cover
            plt.cla()
            plot_arrow(state1.x, state1.y, state1.yaw, fc="m")
            plot_arrow(state2.x, state2.y, state2.yaw, fc="g")
            plot_arrow(state3.x, state3.y, state3.yaw, fc="b")
            # plot_arrow(state4.x, state4.y, state4.yaw, fc="c")

            plt.plot(cx, cy, "-r", label="course")
            plt.plot(cx_short, cy_short, "-y", label="course")

            # plt.plot(x1, y1, "-m", label="trajectory1")
            # plt.plot(x2, y2, "-g", label="trajectory2")
            # plt.plot(x3, y3, "-b", label="trajectory3")
            # plt.plot(x4, y4, "-c", label="trajectory4")

            # plt.plot(cx_short[target_ind1], cy_short[target_ind1], "*", label="target")
            # plt.plot(cx_short[target_ind2], cy_short[target_ind2], "*", label="target")
            # plt.plot(cx_short[target_ind3], cy_short[target_ind3], "*", label="target")
            # plt.plot(cx[target_ind4], cy[target_ind4], "*", label="target")

            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state1.v)[:4])
            plt.pause(0.01) 
            # plt.pause(0.1)       

            # matplotrecorder.save_frame()    

    # matplotrecorder.save_movie("animation.gif", 0.1)

    # assert lastIndex >= target_ind1, "Cannot goal"

if __name__ == '__main__':
    main()

