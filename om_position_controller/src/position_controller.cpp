/*******************************************************************************
* Position Control for 6-DOF Robotic Manipulation
* 
* Description:
* This implementation of Position Control is specifically tailored for a 6-DOF robotic 
* manipulator, focusing on achieving compliant behavior through force feedback control. 
* The code structure and computation methodologies were initially inspired by and adapted 
* from existing gravity compensation strategies employed previously.
*
* Extended from 4-DOF implementation to 6-DOF system.
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



// rosbag record -O trajectory.bag /gravity_compensation_controller/traj_joint_states
// rosbag play trajectory.bag --hz=50

#include "om_position_controller/position_controller.h"

namespace om_position_controller
{
    bool PositionController::init(
    hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle)
{
  // Joint interface initialization
  position_joint_interface_ = robot_hardware->get<hardware_interface::PositionJointInterface>();

  // ROS_INFO_STREAM("Position joint interface" << position_joint_interface_);
  // ROS_INFO(position_joint_interface_); 

  if (position_joint_interface_ == nullptr)
  {
    ROS_ERROR(
        "[PositionController] Could not get position joint interface "
        "from hardware!");
    return false;
  }

  // Joint handle
  if (!node_handle.getParam("joint_names", joint_names_))
  {
    ROS_ERROR("[PositionController] Could not parse joint names");
    return false;
  }
  

  ROS_INFO("%s", std::to_string(joint_names_.size()).c_str());
  // std::vector<float> q_(joint_names_.size(), 0.0); 
  q_.resize(joint_names_.size(), 0.0); 
  ROS_INFO("%s", std::to_string(q_.size()).c_str());
  for (size_t i = 0; i < joint_names_.size(); i++)
  {
    try
    {
        position_joint_handles_.push_back(
            position_joint_interface_->getHandle(joint_names_[i]));    
    }
    catch (...)//(const hardware_interface::HardwareInterfaceException& e)
    {
      ROS_ERROR_STREAM(
          "[PositionController] Could not get joint handle: ");
          // << e.what());
      return false;
    }
  }
  if (q_.size() != position_joint_handles_.size()) {
    ROS_ERROR("q_ and position_joint_handles_ have different sizes!");
    return false; // Exit the update function
  }


//   // Initialize publishers
//   torque_publisher_ = node_handle.advertise<std_msgs::Float64MultiArray>("torque_values", 10);
//   jacobian_publisher_ = node_handle.advertise<std_msgs::Float64MultiArray>("jacobian_values", 10);
//   end_effector_pos_publisher_ = node_handle.advertise<geometry_msgs::Point>("end_effector_position", 10);
//   velocity_publisher_ = node_handle.advertise<std_msgs::Float64MultiArray>("joint_velocities", 10);
//   end_eff_velocity_publisher_ = node_handle.advertise<std_msgs::Float64MultiArray>("end_effector_velocities", 10);
//   end_effector_orient_euler_publisher_ = node_handle.advertise<geometry_msgs::Vector3>("end_effector_orientation", 10);

//   // Optional subscriber to update desired pose at runtime
//   desired_pose_subscriber_ = node_handle.subscribe("desired_pose", 1, &PositionController::desiredPoseCallback, this);
//  desired_pose_subscriber_ = node_handle.subscribe("desired_joint_pose", 1, &PositionController::desiredJointPoseCallback, this);
  // Add a subscriber to listen for desired joint positions
  desired_joint_pos_subscriber_ = node_handle.subscribe("/gravity_compensation_controller//traj_joint_states", 1, &PositionController::desiredJointPosCallback, this);

  // KDL setup
  // urdf::Model urdf;
  // if (!urdf.initParam("robot_description"))
  // {
  //   ROS_ERROR("[PositionController] Could not parse urdf file");
  //   return false;
  // }
  // if (!kdl_parser::treeFromUrdfModel(urdf, kdl_tree_))
  // {
  //   ROS_ERROR("[PositionController] Could not construct kdl tree");
  //   return false;
  // }
  // if (!node_handle.getParam("root_link", root_name_))
  // {
  //   ROS_ERROR("[PositionController] Could not find root link name");
  //   return false;
  // }
  // if (!node_handle.getParam("tip_link", tip_name_))
  // {
  //   ROS_ERROR("[PositionController] Could not find tip link name");
  //   return false;
  // }
  // if (!kdl_tree_.getChain(root_name_, tip_name_, kdl_chain_))
  // {
  //   ROS_ERROR(
  //       "[PositionController] Could not get KDL chain from tree");
  //   return false;
  // }

  // // After setting up the kdl_chain_

  // jnt_to_jac_solver_ = new KDL::ChainJntToJacSolver(kdl_chain_);
  // fk_solver_ = new KDL::ChainFkSolverPos_recursive(kdl_chain_);

  // // Resize the variables for 6 DOF
  // q_.resize(kdl_chain_.getNrOfJoints());
  // tau_.resize(kdl_chain_.getNrOfJoints());
  // G_.resize(kdl_chain_.getNrOfJoints());

  // // Gravity torque setup
  // KDL::Vector g(0.0, 0.0, -9.81); // Adjust based on robot orientation
  // grav_ = g;
  // MCG_solver_.reset(new KDL::ChainDynParam(kdl_chain_, grav_));

  return true;
}

void PositionController::starting(const ros::Time& time) {
  // ROS_INFO("%s", std::to_string(q_.size()).c_str());
  // Initialize joint positions to current values
  for (size_t i = 0; i < joint_names_.size(); i++) {
    // ROS_INFO("Joint %zu has name %s", i, joint_names_[i].c_str());
    // ROS_INFO("Value of q_[0]: %f", q_[0]);
    // ROS_INFO("%f", position_joint_handles_[i].getPosition());    // ROS_INFO("Joint %zu has position %f", i, q_(i));
    q_[i]=position_joint_handles_[i].getPosition();
    // ROS_INFO("%zu", i);
  }
}

void PositionController::update(const ros::Time& time, const ros::Duration& period) {
  // Initialize joint positions to current values
  for (size_t i = 0; i < joint_names_.size(); i++) {
      position_joint_handles_[i].setCommand(q_[i]);
  }
}
void PositionController::stopping(const ros::Time& time) { }

// Callback function for the desired joint positions
    void PositionController::desiredJointPosCallback(const sensor_msgs::JointState::ConstPtr& msg)
    {
        if (msg->position.size() != joint_names_.size()) {
            ROS_ERROR("Received joint position vector size does not match joint count!");
            return;
        }
        
        ROS_INFO("Received desired joint positions");
        for (size_t i = 0; i < joint_names_.size(); i++) {
            q_[i] = msg->position[i];
            ROS_INFO("Setting joint %zu to %f", i, q_[i]);
        }
    }

}

PLUGINLIB_EXPORT_CLASS(
    om_position_controller::PositionController,
    controller_interface::ControllerBase)