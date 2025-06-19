#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import JointState

# Define parameters
num_timesteps = 500      # Total timesteps
total_time = 10.0        # Total trajectory duration (seconds)
dt = total_time / num_timesteps  # Time step interval
t = np.linspace(0, total_time, num_timesteps)  # Time vector

# Initialize trajectory with zeros
desired_positions = np.zeros((num_timesteps, 6))

# Sine wave properties
amplitude = 1.0  # Adjust amplitude if needed
period = 10.0    # Period in seconds
omega = 2 * np.pi / period  # Angular frequency

# Assign sine waves to the last three joints with different phase shifts
desired_positions[:, 3] = amplitude * np.sin(omega * t + 0)         # No delay
desired_positions[:, 4] = amplitude * np.sin(omega * t + np.pi / 3) # Phase shift π/3
desired_positions[:, 5] = amplitude * np.sin(omega * t + 2 * np.pi / 3) # Phase shift 2π/3

# Joint names
joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

def publish_joint_states():
    rospy.init_node('joint_state_publisher', anonymous=True)
    pub = rospy.Publisher('/gravity_compensation_controller//traj_joint_states', JointState, queue_size=10)

    rate = rospy.Rate(int(1/dt))  # Matching the trajectory steps
    
    rospy.loginfo("Publishing joint states...")

    for t_idx in range(num_timesteps):
        if rospy.is_shutdown():
            break

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = joint_names
        msg.position = desired_positions[t_idx, :].tolist()
        msg.velocity = []  # Not needed
        msg.effort = []    # Not needed

        rospy.loginfo("Timestep %d: %s", t_idx, msg.position)
        pub.publish(msg)

        rate.sleep()  # Maintain loop rate

    rospy.loginfo("Finished publishing all timesteps.")

if __name__ == '__main__':
    try:
        publish_joint_states()
    except rospy.ROSInterruptException:
        pass
