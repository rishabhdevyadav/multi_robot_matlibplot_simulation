# multi_agent_matlibplot_sim

MPC
![Alt text](https://github.com/rishabhdevyadav/multi_agent_matlibplot_sim/blob/main/gif/MPC.gif)

PID
![Alt text](https://github.com/rishabhdevyadav/multi_agent_matlibplot_sim/blob/main/gif/PID.gif)

matlibplot simulation

## How To Run

For PID
```bash
python PID_matlibplot.py
```

For MPC
```bash
python MPC_matlibplot.py
```



To save animation commnet and uncomment
```bash
            matplotrecorder.save_frame()    
    matplotrecorder.save_movie("animation.gif", 0.1)
```

Note: in pd_controller parameters are very sensitive. its depend on system processing speed. very hard to tune.
parameter to tune if not working. 

For fernet path parameter at top
```bash
# Parameter for fernet path
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
```


For pd controller in main():
```bash
# Parameter for pd control
gap = 4 #distance between each robot [m]
dt = 0.05
v_ref = 120/3.6
tm.sleep(0.1)

```

```bash
c_speed = 30/3.6
```

c_speed, dt, v_ref, tmsleep() are mutually depend on each other. Need to be tuned simultaneously

### NOTE
Ferenet path planning is trajectory replanning technique.\
"shortpath" abbreviation is technique using local short path slice from full global to reduce computation cost.\
"shortestgap" have a technque of finding desired waypoint for follower robot at fixed distance from leader, but on the same trajectory.\
