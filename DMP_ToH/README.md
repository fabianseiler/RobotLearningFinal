
# Execution of Pick and Place for ToH motions in GAZEBO

## Installation
Copy the `DMP_ToH` folder and `gazebo_ros_link_attacher` to you `catkin_ws/src` folder. Then run ``catkin_make`` and ``devel/setup.bash``.

## Startup Instructions

### 1. Start Link Attacher
First, run:
```

roslaunch gazebo_ros_link_attacher test_attacher.launch

```
This command starts the ROS link attacher, which is necessary for picking up and placing objects in Gazebo.

### 2. Start Simulation Environment with OpenManipulator

```

roslaunch om_position_controller position_control.launch sim:=true

```
This initializes the simulation environment with the OpenManipulator robot.

### 3. Start Cube Publisher
```

roscd om_position_controller/scripts && python3 4_rs_detect_sim.py

```
This script publishes the positions of the cubes via the /tf system, so that the robot can detect and grasp the objects.

### 4. Start Execution Script
```

cd src/execution_scripts

```
```

python3 TOH.py

```
For execution only use
```

python3 TOH_ex.py

```


