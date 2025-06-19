# OpenMANIPULATOR Joint Position Controller 

This package contains the position based controller, which subscribes to the topic  `/gravity_compensation_controller//traj_joint_states` and is intended to be used with the recorded trajectories from the gravity compensation controller. For demonstrating purposes a simple python script was added that publishes one simply crafed trajectory.

## How to run
First make sure all joints are in position based mode and then start the ROS node:
```
roslaunch om_position_controller position_control.launch 
```

Then you could start the replay of the trajectory recorded:
```
rosbag play trajectory.bag --hz=50
```

or start the python script for publishing the trajectory:
```
roscd om_position_controller && cd scripts && chmod +x publish_joint_positions.py && python publish_joint_positions.py
```
