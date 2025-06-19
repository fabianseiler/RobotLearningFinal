#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

class JointStateRepublisher:
    def __init__(self):
        rospy.init_node('joint_trajectory_publisher', anonymous=True)
        
        # Publisher to arm controller
        self.pub = rospy.Publisher('/open_manipulator_6dof/arm_controller/command', JointTrajectory, queue_size=10)
        
        # Subscriber to gravity compensation controller
        self.sub = rospy.Subscriber('/gravity_compensation_controller/traj_joint_states', JointState, self.joint_state_callback)
        
        # Store latest joint state
        self.latest_joint_state = None
        
        rospy.loginfo("Joint state republisher initialized")
        
    def joint_state_callback(self, msg):
        """Callback function to receive joint states and republish as trajectory"""
        self.latest_joint_state = msg
        
        # Extract only the arm joints (first 6 joints, excluding gripper)
        if len(msg.position) >= 6:
            arm_positions = msg.position[:6]
            arm_joint_names = msg.name[:6]
            
            # Create trajectory message
            traj_msg = JointTrajectory()
            traj_msg.header.stamp = rospy.Time.now()
            traj_msg.joint_names = arm_joint_names
            
            # Create a single trajectory point with current positions
            point = JointTrajectoryPoint()
            point.positions = list(arm_positions)
            point.time_from_start = rospy.Duration(0.1)  # Small duration for immediate execution
            traj_msg.points.append(point)
            
            # Publish the trajectory
            self.pub.publish(traj_msg)
            
            rospy.loginfo("Republished joint positions: %s", arm_positions)

def main():
    try:
        republisher = JointStateRepublisher()
        rospy.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        rospy.loginfo("Joint state republisher shutting down")

if __name__ == '__main__':
    main()