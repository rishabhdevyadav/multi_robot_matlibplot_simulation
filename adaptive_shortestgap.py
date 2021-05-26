#!/usr/bin/python
#PID Trajectory tracking
# import time
import time as tm
import math

import matplotlib.pyplot as plt
import numpy as np
# import scipy.linalg as la


import matplotrecorder
import cubic_spline_planner


show_animation = True
old_nearest_point_index = None


Kp01, Kp11, des_v1  = 1,1,1
Kp02, Kp12, des_v2  = 1,1,1
Kp03, Kp13, des_v3  = 1,1,1

Kq01, Kq11, des_omega1 = 1, 1, 0
Kq02, Kq12, des_omega2 = 1, 1, 0
Kq03, Kq13, des_omega3 = 1, 1, 0



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
        # print(t)

        x_des = (1 - t)*x0 + t*x1
        y_des = (1 - t)*y0 + t*y1

        return x_des, y_des, t

def  agent_angSpeed_Dpoint(state, des_x, des_y):
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

    return (error)*0.01


def dist_error(L_state, F_state, targDist):
    dist = np.sqrt( np.square(L_state.x - F_state.x) + np.square(L_state.y - F_state.y))
    error = (targDist - dist)
    return error



def agent_adaptive_linSpeed1(v_ref, curr_v, L_error, F_error, dt):
    global Kp01, Kp11, des_v1
    Phi = 0.25
    # errPos = err12
    errPos = (L_error + F_error )

    errVel = curr_v - v_ref
    sv = errVel + Phi*errPos

    alpha_0, alpha_1 = 5, 5
    Kp01 += (sv - alpha_0*Kp01)*dt
    Kp11 += (sv - alpha_1*Kp11)*dt
    # Kp0 = np.maximum(Kp0, 0.001)
    # Kp1 = np.maximum(Kp1, 0.001)

    Rho = Kp01 + Kp11*errPos

        # v = 0.1
        # if np.absolute(sv) > v:
        #     deltau = Rho*(sv/np.absolute(sv))
        # else:
        #     deltau = Rho*(sv/v)

    deltau = Rho*(sv/np.absolute(sv))

    Lamd = 1
    des_accel = -Lamd*sv - deltau
    des_v1 += des_accel*dt  

    return des_v1


def agent_adaptive_linSpeed2(v_ref, curr_v, L_error, F_error, dt):
    global Kp02, Kp12, des_v2
    Phi = 0.25
    # errPos = err12
    errPos = (L_error + F_error )

    errVel = curr_v - v_ref
    sv = errVel + Phi*errPos

    alpha_0, alpha_1 = 0.5, 0.5
    Kp02 += (sv - alpha_0*Kp02)*dt
    Kp12 += (sv - alpha_1*Kp12)*dt
    # Kp0 = np.maximum(Kp0, 0.001)
    # Kp1 = np.maximum(Kp1, 0.001)

    Rho = Kp02 + Kp12*errPos

        # v = 0.1
        # if np.absolute(sv) > v:
        #     deltau = Rho*(sv/np.absolute(sv))
        # else:
        #     deltau = Rho*(sv/v)

    deltau = Rho*(sv/np.absolute(sv))

    Lamd = 1
    des_accel = -Lamd*sv - deltau
    des_v2 += des_accel*dt  

    return des_v2


def agent_adaptive_linSpeed3(v_ref, curr_v, L_error, F_error, dt):
    global Kp03, Kp13, des_v3
    Phi = 0.25
    # errPos = err12
    errPos = (L_error + F_error )

    errVel = curr_v - v_ref
    sv = errVel + Phi*errPos

    alpha_0, alpha_1 = 0.5, 0.5
    Kp03 += (sv - alpha_0*Kp03)*dt
    Kp13 += (sv - alpha_1*Kp13)*dt
    # Kp0 = np.maximum(Kp0, 0.001)
    # Kp1 = np.maximum(Kp1, 0.001)

    print(Kp03, Kp13)
    print("----------")

    Rho = Kp03 + Kp13*errPos

        # v = 0.1
        # if np.absolute(sv) > v:
        #     deltau = Rho*(sv/np.absolute(sv))
        # else:
        #     deltau = Rho*(sv/v)

    deltau = Rho*(sv/np.absolute(sv))

    Lamd = 1
    des_accel = -Lamd*sv - deltau
    des_v3 += des_accel*dt  

    return des_v3


def agent_adaptive_angSpeed1(omega_ref, curr_omega, omega_error, dt):
    # dt = 0.01
    global Kq01, Kq11, des_omega1
    errPosq = omega_error
    svq = errPosq
    alphaq_0, alphaq_1 = 5, 5
    Kq01 += (svq - alphaq_0*Kq01)*dt
    Kq11 += (svq - alphaq_1*Kq11)*dt

    Rhoq = Kq01 + Kq11*errPosq

    if svq == 0.0:
        deltauq = 0
    else:
        deltauq = Rhoq*(svq/np.absolute(svq))

    Lamdq = 10
    des_omega1 = Lamdq*svq + deltauq

    return des_omega1

def agent_adaptive_angSpeed2(omega_ref, curr_omega, omega_error, dt):
    # dt = 0.01
    global Kq02, Kq12, des_omega2
    Phiq = 0.6
    errPosq = omega_error
    svq = errPosq

    alphaq_0, alphaq_1 = 5, 5
    Kq02 += (svq - alphaq_0*Kq02)*dt
    Kq12 += (svq - alphaq_1*Kq12)*dt

    Rhoq = Kq02 + Kq12*errPosq

    if svq == 0.0:
        deltauq = 0
    else:
        deltauq = Rhoq*(svq/np.absolute(svq))

    Lamdq = 10
    des_omega2 = Lamdq*svq + deltauq

    return des_omega2

def agent_adaptive_angSpeed3(omega_ref, curr_omega, omega_error, dt):
    # dt = 0.01
    global Kq03, Kq13, des_omega3
    Phiq = 0.6
    errPosq = omega_error

    svq = errPosq

    alphaq_0, alphaq_1 = 5, 5
    Kq03 += (svq - alphaq_0*Kq03)*dt
    Kq13 += (svq - alphaq_1*Kq13)*dt

    Rhoq = Kq03 + Kq13*errPosq

    if svq == 0.0:
        deltauq = 0
    else:
        deltauq = Rhoq*(svq/np.absolute(svq))

    Lamdq = 10
    des_omega3 = Lamdq*svq + deltauq 

    return des_omega3


def main():
    #ax = [0,10,20,30,40,50,60,70,80]  ##still creating few problem
    # ax = [-15, -10.0, -5.0,  0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0 ]
    # ay = [0.0,   0.0,  0.0,  0.0, 3.0,  0.0, -3.0,  0.0,  0.0,  0.0]
    ax = [-20, -10, -5.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,100 ]
    ay = [0.0, 0.0, 0.0, 0.0, 5.0,  0.0, -5.0,  0.0,  0.0,  0.0, 10.0]
    #ay = [math.sin(ix / 5.0) * ix / 2.0 for ix in ax]
    goal = [ax[-1], ay[-1]]
    cx, cy, cyaw, ck_curv, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    ck = np.absolute(ck_curv) * [10]
    
    lastIndex = len(cx) - 1
    T, time, t = 100, 0.0, [0]
    gap = 6

    state1 = State(x=-5, y=0, yaw=0, v=0.0,omega=0.0)
    state2 = State(x=-11, y=0, yaw=0, v=0.0,omega=0.0)
    state3 = State(x=-17, y=0, yaw=0, v=0.0,omega=0.0)
    # state4 = State(x=-15, y=0, yaw=0, v=0.0,omega=0.0)


    x1, y1, yaw1, v1, omega1 = [state1.x], [state1.y], [state1.yaw], [state1.v], [state1.omega]
    x2, y2, yaw2, v2, omega2 = [state2.x], [state2.y], [state2.yaw], [state2.v], [state2.omega]
    x3, y3, yaw3, v3, omega3 = [state3.x], [state3.y], [state3.yaw], [state3.v], [state3.omega]
    # x4, y4, yaw4, v4, omega4 = [state4.x], [state4.y], [state4.yaw], [state4.v], [state4.omega]

    
    v_in1, v_in2, v_in3 = 9.5, 9.5, 9.5
    omega_adap1, omega_adap2, omega_adap3 = 0, 0, 0

    start_time = tm.time()
    last_time = start_time

    while True :

        current_time = tm.time()
        dt = (current_time - last_time)
        # print(dt)

        dt = 0.05
        v_ref = 10


        #state1 omega calculate
        target_ind1 = calc_target_index(state1, cx, cy) 
        des_x1 = cx[target_ind1]
        des_y1 = cy[target_ind1]
        AngErr1 = agent_angSpeed_Dpoint(state1, des_x1, des_y1)
        omega_adap1 = agent_adaptive_angSpeed1(0, omega_adap1, AngErr1, dt)

        

        #state2 omega calculate
        des_x2, des_y2, t2 = follwer_target(state1, state2, gap)
        AngErr2 =  agent_angSpeed_Dpoint(state2, des_x2, des_y2)
        omega_adap2 = agent_adaptive_angSpeed2(0, omega_adap2, AngErr2, dt)
        

        #state3 omega calculate
        des_x3, des_y3, t3 = follwer_target(state2, state3, gap)
        AngErr3 =  agent_angSpeed_Dpoint(state3, des_x3, des_y3)
        omega_adap3 = agent_adaptive_angSpeed3(0, omega_adap3, AngErr3, dt)
        

        #agent linear speed
        err12 = dist_error(state1, state2, gap)
        err23 = dist_error(state2, state3, gap)


        v_in1 = agent_adaptive_linSpeed1(v_ref, v_in1, 0, 0, dt)
        v_in2 = agent_adaptive_linSpeed2(v_ref, v_in2, err12, 0, dt)
        v_in3 = agent_adaptive_linSpeed3(v_ref, v_in3, err23, 0, dt)



        # v_in1 = 10
        # v_in2 = 10
        # v_in3 = 10

        #clipping of speed; if follower is too close then stop it
        if t2 > 1 : 
            v_in2 = 0
            omega_adap2 = 0.0
            # print("here t2")
        
        if t3 > 1 : 
            v_in3 = 0
            omega_adap3 = 0.0
            # print("here t3")

        # print(omega_adap1, omega_adap2, omega_adap3)
        print(v_in1, v_in2, v_in3)

        # print(err12, err23)

        #agent state update
        state1 = agent_state_update(state1, v_in1, omega_adap1, dt)
        state2 = agent_state_update(state2, v_in2, omega_adap2, dt)
        state3 = agent_state_update(state3, v_in3, omega_adap3, dt)

        last_time = current_time
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

            plot_arrow(state1.x, state1.y, state1.yaw, fc="m")
            plot_arrow(state2.x, state2.y, state2.yaw, fc="g")
            plot_arrow(state3.x, state3.y, state3.yaw, fc="b")
            # plot_arrow(state4.x, state4.y, state4.yaw, fc="c")

            # plt.plot(tx, ty)
            plt.plot(cx, cy, "-r", label="course")
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

