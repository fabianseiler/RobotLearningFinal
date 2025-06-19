#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import copy
from sensor_msgs.msg import JointState

class HomeMoverJointStates:
    def __init__(self):
        rospy.init_node("home_mover_joint_states", anonymous=True)

        self.joint_names = ["gripper", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.home_position = np.array([-0.03834952, -1, 1.26093221, 0.00613592, 1.825, -0.00460194])
        #[-0.02761165425181389, -0.6135923266410828, 1.4035924673080444, 0.02147573232650757, 1.4649516344070435, 0.00920388475060463]
        self.current_position = None

        self.sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        self.pub = rospy.Publisher("/gravity_compensation_controller/traj_joint_states", JointState, queue_size=10)

        self.rate = rospy.Rate(20)  # 20 Hz

        rospy.loginfo("Waiting for joint states...")
        while self.current_position is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Joint states received. Ready to move.")

    def joint_state_callback(self, msg):
        if set(self.joint_names).issubset(set(msg.name)):
            name_to_pos = dict(zip(msg.name, msg.position))
            self.current_position = [name_to_pos[name] for name in self.joint_names]
            #print(self.current_position)

    def move_to_home(self, steps=100, duration=5.0):
        rate = rospy.Rate(steps / duration)

        #start = np.array(copy.deepcopy(self.current_position))
        while (self.current_position is None or 
               np.allclose(self.current_position[:6], np.zeros(6))) and not rospy.is_shutdown():
            rospy.loginfo_throttle(1, "Waiting for valid joint positions...")
            rospy.sleep(0.1)
        start = np.array(copy.deepcopy(self.current_position))
        start_gripper = start[0]  # gripper position
        start_joints = start[1:]  # joint1-6
        goal_joints = self.home_position  # Nur joint1-6

        for i in range(1, steps + 1):
            alpha = i / float(steps)
            interp_joints = (1 - alpha) * start_joints + alpha * goal_joints

            # Position: joints zuerst, dann gripper
            reordered_position = interp_joints.tolist() + [start_gripper]
            reordered_names = self.joint_names[1:] + [self.joint_names[0]]  # joint1-6, gripper

            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.name = reordered_names
            msg.position = reordered_position
            msg.velocity = [0.0] * len(self.joint_names)
            msg.effort = [0.0] * len(self.joint_names)

            self.pub.publish(msg)
            rate.sleep()

        rospy.loginfo("Home position reached.")



if __name__ == '__main__':
    try:
        mover = HomeMoverJointStates()
        rospy.sleep(2.0)
        mover.move_to_home()
    except rospy.ROSInterruptException:
        pass
