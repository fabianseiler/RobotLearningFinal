/*******************************************************************************
* Copyright 2020 ROBOTIS CO., LTD.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/* Authors: Ryan Shim */
/* Impedance Control adaptation: Debojit Das, Extended for 6-DOF */

#ifndef ADVANCED_ROBOTICS_CONTROL_IMPEDANCE_CONTROL_CONTROLLER_H
#define ADVANCED_ROBOTICS_CONTROL_IMPEDANCE_CONTROL_CONTROLLER_H

#include <boost/scoped_ptr.hpp>
#include <string>
#include <vector>
#include <algorithm> // For std::min, std::max
#include <cmath>     // For mathematical functions

#include <controller_interface/controller.h>
#include <controller_interface/controller_base.h>
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <kdl/tree.hpp>
#include <kdl/kdl.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chain.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/jacobian.hpp>
#include <kdl/jntarray.hpp>
#include <pluginlib/class_list_macros.h>
#include <ros/node_handle.h>
#include <ros/ros.h>
#include <ros/time.h>
#include <urdf/model.h>
#include <std_msgs/Float64MultiArray.h> // For publishing torque values
#include <kdl/chainjnttojacsolver.hpp> // For Jacobian calculation
#include <kdl/chainfksolverpos_recursive.hpp> // For forward kinematics
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/MultiArrayDimension.h>
#include <kdl/chainfksolvervel_recursive.hpp>
#include <sensor_msgs/JointState.h>


namespace om_position_controller
{
class PositionController
    : public controller_interface::MultiInterfaceController<
          hardware_interface::PositionJointInterface>
{
 public:
  bool init(hardware_interface::RobotHW* robot_hardware,
            ros::NodeHandle& node_handle) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void stopping(const ros::Time& time) override;

 private:
  // Joint handle
  hardware_interface::PositionJointInterface* position_joint_interface_;
  std::vector<hardware_interface::JointHandle> position_joint_handles_;
  std::vector<std::string> joint_names_;
  std::string root_name_;
  std::string tip_name_;
  std::vector<float> q_;

  // KDL

  // KDL solvers
  boost::scoped_ptr<KDL::ChainDynParam> MCG_solver_;
  KDL::ChainJntToJacSolver* jnt_to_jac_solver_;
  KDL::ChainFkSolverPos_recursive* fk_solver_;
  
  // Publishers
  ros::Publisher end_effector_pos_publisher_;
  ros::Publisher end_effector_orient_euler_publisher_;
  ros::Publisher torque_publisher_;
  ros::Publisher jacobian_publisher_;
  ros::Publisher velocity_publisher_;
  ros::Publisher end_eff_velocity_publisher_;
  
  // Subscriber for potential desired pose updates
  ros::Subscriber desired_joint_pos_subscriber_;
  
  // Callback for updating desired pose
  void desiredJointPosCallback(const sensor_msgs::JointState::ConstPtr& msg);
};
}  // namespace advanced_robotics_control
#endif  // ADVANCED_ROBOTICS_CONTROL_IMPEDANCE_CONTROL_CONTROLLER_H