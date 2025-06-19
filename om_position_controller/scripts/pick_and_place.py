import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.trajectories as ptr
from movement_primitives.kinematics import Kinematics
import rosbag
from tf.transformations import quaternion_matrix
from movement_primitives.dmp import CartesianDMP
import pickle
import os
import time
from scipy.interpolate import interp1d
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import tf
import matplotlib.pyplot as plt

class DMPMotionGenerator:
    def __init__(self, urdf_path, mesh_path=None, joint_names=None, base_link="world", end_effector_link="end_effector_link"):
        print("Initializing DMPMotionGenerator for Gazebo...")
        self.urdf_path = urdf_path
        self.mesh_path = mesh_path
        self.kin = self._load_kinematics(urdf_path, mesh_path)
        
        self.joint_names = joint_names or ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_joint_names = ["gripper", "gripper_sub"]
        self.base_link = base_link
        self.end_effector_link = end_effector_link
        self.chain = self.kin.create_chain(self.joint_names, base_link, end_effector_link)
        self.dmp = None
        self.IK_joint_trajectory = None
        self.gripper_trajectory = None
        
        if not rospy.core.is_initialized():
            rospy.init_node('dmp_motion_generator', anonymous=True)

    def _load_kinematics(self, urdf_path, mesh_path=None):
        with open(urdf_path, 'r') as f:
            return Kinematics(f.read(), mesh_path=mesh_path)

    def learn_from_rosbag(self, bag_path, joint_topic, dt=None, n_weights=10):
        transforms, joint_trajectory, gripper_trajectory, time_stamp = self._process_rosbag(bag_path, joint_topic)
        self.gripper_trajectory = gripper_trajectory
        
        print(f"Transforms shape: {transforms.shape}")
        Y = ptr.pqs_from_transforms(transforms[10:,:,:])
        if dt is None:
            dt = 1/self.frequency
        self.dmp = CartesianDMP(execution_time=max(time_stamp), dt=dt, n_weights_per_dim=n_weights)
        self.dmp.imitate(time_stamp[10:], Y)
        
        return Y, transforms, joint_trajectory, gripper_trajectory

    def _process_rosbag(self, bag_path, joint_topic):
        transforms = []
        joint_trajectory = []
        gripper_trajectory = []
        time_stamp = []
        
        print(f"Reading bag file: {bag_path}")
        bag = rosbag.Bag(bag_path)
        for topic, msg, t in bag.read_messages(topics=[joint_topic]):
            joint_pos = msg.position[:6]
            gripper_pos = msg.position[6] if len(msg.position) > 6 else 0.0
            joint_trajectory.append(joint_pos)
            gripper_trajectory.append(gripper_pos)

            transforms.append(self.chain.forward(joint_pos))
            time_stamp.append(msg.header.stamp.to_sec())    
        bag.close()
        
        transforms = np.array(transforms)
        joint_trajectory = np.array(joint_trajectory)
        gripper_trajectory = np.array(gripper_trajectory)
        time_stamp = np.array(time_stamp)
        
        dt = []
        for i in range(1, time_stamp.shape[0]):
            dt.append(time_stamp[i]- time_stamp[i-1])
        self.frequency = 1/ np.average(np.array(dt))
        
        positions = np.array([T[:3, 3] for T in transforms])
        mask, _ = self.remove_outliers_mad(positions, threshold=5.0)
        
        filtered_time = time_stamp[mask]
        normalized_time = filtered_time - filtered_time[0]
        
        return transforms[mask], joint_trajectory[mask], gripper_trajectory[mask], normalized_time

    def remove_outliers_mad(self, data, threshold=3.5):
        median = np.median(data, axis=0)
        diff = np.abs(data - median)
        mad = np.median(diff, axis=0)
        modified_z_score = 0.6745 * diff / (mad + 1e-6)
        mask = np.all(modified_z_score < threshold, axis=1)
        return mask, data[mask]

    def generate_trajectory(self, start_y=None, goal_y=None):
        print(f"Generating trajectory")
        if self.dmp is None:
            raise ValueError("No DMP model available. Learn or load a model first.")
            
        if start_y is not None:
            self.dmp.start_y = start_y
            print(f"Using custom start: {start_y}")
        else:
            print(f"Using default start: {self.dmp.start_y}")
            
        if goal_y is not None:
            self.dmp.goal_y = goal_y
            print(f"Using custom goal: {goal_y}")
        else:
            print(f"Using default goal: {self.dmp.goal_y}")
        
        T, Y = self.dmp.open_loop()
        trajectory = ptr.transforms_from_pqs(Y)
        return T, trajectory

    def save_dmp(self, filepath):
        if self.dmp is None:
            rospy.logerr("No DMP model available to save.")
            return
        if self.gripper_trajectory is None:
            rospy.logwarn("Gripper trajectory not available or not learned. Saving None for gripper_trajectory.")

        data_to_save = {
            'dmp': self.dmp,
            'gripper_trajectory': self.gripper_trajectory
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            rospy.loginfo(f"DMP and gripper trajectory saved to {filepath}")
        except Exception as e:
            rospy.logerr(f"Failed to save DMP data to {filepath}: {e}")

    def load_dmp(self, filepath):
        rospy.loginfo(f"Loading DMP data from {filepath}")
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)

            if isinstance(loaded_data, dict):
                if 'dmp' in loaded_data:
                    self.dmp = loaded_data['dmp']
                else:
                    rospy.logerr("Loaded dictionary is missing 'dmp' key.")
                    self.dmp = None

                if 'gripper_trajectory' in loaded_data:
                    self.gripper_trajectory = loaded_data['gripper_trajectory']
                    if self.gripper_trajectory is not None:
                         rospy.loginfo(f"Gripper trajectory loaded ({len(self.gripper_trajectory)} points).")
                    else:
                         rospy.loginfo("Loaded None for gripper trajectory.")
                else:
                    rospy.logwarn("Loaded dictionary is missing 'gripper_trajectory' key. Setting to None.")
                    self.gripper_trajectory = None
            else:
                rospy.logwarn("Loading old DMP format (only DMP object found). Gripper trajectory will be None.")
                self.dmp = loaded_data
                self.gripper_trajectory = None

            if self.dmp:
                rospy.loginfo("DMP object loaded successfully.")
            else:
                 rospy.logerr("Failed to load DMP object.")

        except FileNotFoundError:
            rospy.logerr(f"DMP file not found: {filepath}")
            self.dmp = None
            self.gripper_trajectory = None
        except Exception as e:
            rospy.logerr(f"Error loading DMP data from {filepath}: {e}")
            self.dmp = None
            self.gripper_trajectory = None
    
    def compute_IK_trajectory(self, trajectory, time_stamp, q0=None, subsample_factor=1):
        if q0 is None:
            q0 = np.array([-0.03834952, -0.84062147, 1.26093221, 0.00613592, 1.97576725, -0.00460194])
        
        if subsample_factor > 1:
            subsampled_trajectory = trajectory[::subsample_factor]
            subsampled_time_stamp = time_stamp[::subsample_factor]
            subsampled_gripper_trajectory = self.gripper_trajectory[::subsample_factor] if self.gripper_trajectory is not None else None
            print(f"Subsampled time from {len(time_stamp)} to {len(subsampled_time_stamp)} points")
            print(f"Subsampled trajectory from {len(trajectory)} to {len(subsampled_trajectory)} points")
        else:
            subsampled_trajectory = trajectory
            subsampled_time_stamp = time_stamp
            subsampled_gripper_trajectory = self.gripper_trajectory
        
        print(f"Solving inverse kinematics for {len(subsampled_trajectory)} points...")
        
        start_time = time.time()
        
        random_state = np.random.RandomState(0)
        joint_trajectory = self.chain.inverse_trajectory(
            subsampled_trajectory, random_state=random_state, orientation_weight=1.0)
            
        print(f"IK solved in {time.time() - start_time:.2f} seconds")
        
        return subsampled_trajectory, joint_trajectory, subsampled_gripper_trajectory, subsampled_time_stamp

    def visualize_trajectory(self, trajectory, joint_trajectory, q0=None):
        print(f"Plotting trajectory...")
        fig = pv.figure()
        fig.plot_transform(s=0.3)
        
        graph = fig.plot_graph(
            self.kin.tm, "world", show_visuals=False, show_collision_objects=True,
            show_frames=True, s=0.1, whitelist=[self.base_link, self.end_effector_link])

        fig.plot_transform(trajectory[0], s=0.15)
        fig.plot_transform(trajectory[-1], s=0.15)
        
        pv.Trajectory(trajectory, s=0.05).add_artist(fig)
        
        fig.view_init()
        fig.animate(
            animation_callback, len(trajectory), loop=True,
            fargs=(graph, self.chain, joint_trajectory))
        fig.show()


class GazeboTrajectoryPublisher:
    def __init__(self, joint_names=None, gripper_joint_names=None):
        if not rospy.core.is_initialized():
            rospy.init_node("gazebo_trajectory_publisher", anonymous=True)
        
        self.joint_names = joint_names or ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_joint_names = gripper_joint_names or ["gripper", "gripper_sub"]
        
        self.arm_pub = rospy.Publisher('/open_manipulator_6dof/arm_controller/command', 
                                     JointTrajectory, queue_size=10)
        self.gripper_pub = rospy.Publisher('/open_manipulator_6dof/gripper_controller/command', 
                                         JointTrajectory, queue_size=10)
        
        print(f"[Gazebo] Initialized publishers:")
        print(f"  - Arm: /open_manipulator_6dof/arm_controller/command")
        print(f"  - Gripper: /open_manipulator_6dof/gripper_controller/command")
        
        rospy.sleep(1.0)

    def publish_trajectory(self, joint_trajectory, gripper_trajectory, timestamps, execute_time_factor=1.0):
        if len(joint_trajectory) == 0:
            rospy.logwarn("[Gazebo] Empty trajectory provided")
            return
        
        print(f"[Gazebo] Publishing trajectory with {len(joint_trajectory)} points")
        
        arm_msg = JointTrajectory()
        arm_msg.header.stamp = rospy.Time.now()
        arm_msg.joint_names = self.joint_names
        
        gripper_msg = JointTrajectory()
        gripper_msg.header.stamp = rospy.Time.now()
        gripper_msg.joint_names = self.gripper_joint_names
        
        for i in range(len(joint_trajectory)):
            arm_point = JointTrajectoryPoint()
            arm_point.positions = joint_trajectory[i].tolist()
            arm_point.velocities = [0.0] * len(self.joint_names)
            arm_point.accelerations = [0.0] * len(self.joint_names)
            arm_point.time_from_start = rospy.Duration.from_sec(
                (timestamps[i] - timestamps[0]) * execute_time_factor
            )
            arm_msg.points.append(arm_point)
            
            if gripper_trajectory is not None and i < len(gripper_trajectory):
                gripper_point = JointTrajectoryPoint()
                gripper_value = gripper_trajectory[i]
                gripper_point.positions = [-2.0*gripper_value, -2.0*gripper_value]
                gripper_point.velocities = [0.0, 0.0]
                gripper_point.accelerations = [0.0, 0.0]
                gripper_point.time_from_start = rospy.Duration.from_sec(
                    (timestamps[i] - timestamps[0]) * execute_time_factor
                )
                gripper_msg.points.append(gripper_point)
        
        print(f"[Gazebo] Publishing arm trajectory with {len(arm_msg.points)} points")
        self.arm_pub.publish(arm_msg)
        
        if gripper_trajectory is not None and len(gripper_msg.points) > 0:
            print(f"[Gazebo] Publishing gripper trajectory with {len(gripper_msg.points)} points")
            self.gripper_pub.publish(gripper_msg)
        else:
            print(f"[Gazebo] No gripper trajectory to publish")
        
        print(f"[Gazebo] Trajectory published successfully")

    def publish_single_trajectory(self, full_trajectory, timestamps, execute_time_factor=1.0):
        if full_trajectory.shape[1] >= 6:
            arm_traj = full_trajectory[:, :6]
            gripper_traj = full_trajectory[:, 6] if full_trajectory.shape[1] > 6 else None
            
            self.publish_trajectory(arm_traj, gripper_traj, timestamps, execute_time_factor)
        else:
            rospy.logwarn(f"[Gazebo] Invalid trajectory shape: {full_trajectory.shape}")

    def publish_home_position(self, home_position=None, execution_time=5.0):
        if home_position is None:
            home_position = [-0.03834952, -0.84062147, 1.26093221, 0.00613592, 1.97576725, -0.00460194]
        
        print(f"[Gazebo] Publishing home position command...")
        print(f"[Gazebo] Home position: {home_position}")
        print(f"[Gazebo] Execution time: {execution_time} seconds")
        
        arm_msg = JointTrajectory()
        arm_msg.header.stamp = rospy.Time.now()
        arm_msg.joint_names = self.joint_names
        
        home_point = JointTrajectoryPoint()
        home_point.positions = home_position
        home_point.velocities = [0.0] * len(self.joint_names)
        home_point.accelerations = [0.0] * len(self.joint_names)
        home_point.time_from_start = rospy.Duration.from_sec(execution_time)
        
        arm_msg.points.append(home_point)
        
        self.arm_pub.publish(arm_msg)
        print(f"[Gazebo] Home position command published and latched")


def animation_callback(step, graph, chain, joint_trajectory):
    chain.forward(joint_trajectory[step])
    graph.set_data()
    return graph

def save_trajectory_data(joint_trajectory, timestamps, filepath):
    data = {
        'trajectory': joint_trajectory,
        'timestamps': timestamps
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"[SAVE] Trajectory data saved to {filepath}")

def load_trajectory_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    joint_trajectory = data['trajectory']
    timestamps = data['timestamps']
    print(f"[LOAD] Loaded trajectory from {filepath} (length={len(joint_trajectory)})")
    return joint_trajectory, timestamps

def interpolate_joint_trajectory(joint_traj, time_stamps, target_freq=20.0):
    num_joints = joint_traj.shape[1]
    duration = time_stamps[-1] - time_stamps[0]
    num_samples = int(duration * target_freq)
    new_timestamps = np.linspace(time_stamps[0], time_stamps[-1], num_samples)
    
    interp_traj = np.zeros((num_samples, num_joints))
    for i in range(num_joints):
        interpolator = interp1d(time_stamps, joint_traj[:, i], kind='linear', fill_value="extrapolate")
        interp_traj[:, i] = interpolator(new_timestamps)
    
    return interp_traj, new_timestamps

def get_cube_position(cube_name, timeout=5.0):
    """Get position of a cube by its TF frame name"""
    print(f"Getting {cube_name} position...")
    if not rospy.core.is_initialized():
        rospy.init_node('tf_xyz_fetcher', anonymous=True)
    
    listener = tf.TransformListener()
    try:
        print(f"Waiting for transform /world -> /{cube_name}...")
        listener.waitForTransform('/world', f'/{cube_name}', rospy.Time(0), rospy.Duration(timeout))
        
        trans, _ = listener.lookupTransform('/world', f'/{cube_name}', rospy.Time(0))
        print(f"{cube_name} position: {trans}")
        return trans
    except Exception as e:
        print(f"Error getting transform for {cube_name}: {e}")
        return None

def execute_motion(dmp_gen, bag_path, dmp_save_path, cube_name, position_offset, publisher, 
                  motion_name="motion", execute_time_factor=5, visualize=False):
    """Execute a motion (pick or place) with given parameters"""
    print(f"\n=== Executing {motion_name} motion ===")
    
    # Learn from bag
    print(f"Learning {motion_name} motion from bag: {bag_path}")
    Y, transforms, joint_traj, gripper_traj = dmp_gen.learn_from_rosbag(
        bag_path, 
        '/gravity_compensation_controller/traj_joint_states'
    )
    
    # Save DMP
    dmp_gen.save_dmp(dmp_save_path)
    print(f"{motion_name.capitalize()} DMP saved to: {dmp_save_path}")
    
    # Get target position
    cube_position = get_cube_position(cube_name)
    if cube_position is None:
        print(f"Failed to get {cube_name} position. Using default offset.")
        cube_position = [0.0, 0.0, 0.0]
    
    # Set start and goal
    new_start = dmp_gen.dmp.start_y.copy()
    new_goal = dmp_gen.dmp.goal_y.copy()
    
    print(f"Original goal: {new_goal[:3]}")
    new_goal[:3] = np.array(cube_position) + np.array(position_offset)
    print(f"New goal: {new_goal[:3]}")
    
    # Generate trajectory
    T, trajectory = dmp_gen.generate_trajectory(start_y=new_start, goal_y=new_goal)
    
    # Compute IK
    trajectory, IK_joint_trajectory, gripper_traj, T = dmp_gen.compute_IK_trajectory(
        trajectory, T, subsample_factor=1)
    
    # Apply smoothing
    window_size = 25
    if len(IK_joint_trajectory) > window_size:
        original_start = IK_joint_trajectory[0,:].copy()
        original_end = IK_joint_trajectory[-1,:].copy()

        smoothed_IK_joint_trajectory = np.zeros_like(IK_joint_trajectory)
        for i in range(IK_joint_trajectory.shape[1]):
            smoothed_IK_joint_trajectory[:, i] = np.convolve(IK_joint_trajectory[:, i], 
                                                           np.ones(window_size)/window_size, mode='same')

        smoothed_IK_joint_trajectory[0,:] = original_start
        smoothed_IK_joint_trajectory[-1,:] = original_end

        half_window = window_size // 2
        for i in range(IK_joint_trajectory.shape[1]):
            for j in range(half_window):
                alpha = j / float(half_window)
                smoothed_IK_joint_trajectory[j, i] = (1 - alpha) * original_start[i] + alpha * smoothed_IK_joint_trajectory[j, i]
            for j in range(half_window):
                alpha = j / float(half_window)
                idx_from_end = len(IK_joint_trajectory) - 1 - j
                smoothed_IK_joint_trajectory[idx_from_end, i] = (1 - alpha) * original_end[i] + alpha * smoothed_IK_joint_trajectory[idx_from_end, i]

        IK_joint_trajectory = smoothed_IK_joint_trajectory
        print(f"Applied moving average filter with window size {window_size} to IK trajectory.")
    else:
        print(f"Trajectory too short for smoothing (length {len(IK_joint_trajectory)})")

    # Visualize if requested
    
    if visualize:
        dmp_gen.visualize_trajectory(trajectory, IK_joint_trajectory)
    
    # Prepare full trajectory
    traj_length = min(IK_joint_trajectory.shape[0], len(gripper_traj) if gripper_traj is not None else IK_joint_trajectory.shape[0])
    IK_joint_trajectory = IK_joint_trajectory[:traj_length, :]
    
    if gripper_traj is not None:
        gripper_traj = gripper_traj[:traj_length]
        full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))
    else:
        gripper_traj = np.zeros(traj_length)
        full_trajectory = np.hstack((IK_joint_trajectory, -gripper_traj.reshape(-1, 1)))
    
    # Interpolate trajectory
    interpolated_traj, interpolated_time = interpolate_joint_trajectory(
        full_trajectory, T[:traj_length], target_freq=100.0)

    # Execute trajectory
    print(f"[{motion_name}] Starting trajectory execution...")
    
    # Clip trajectory to 95% of length to avoid oscillations in the learned motions
    clip_length = int(0.95 * len(interpolated_traj))
    arm_trajectory = interpolated_traj[:clip_length, :6]
    gripper_trajectory = interpolated_traj[:clip_length, 6]
    
    publisher.publish_trajectory(arm_trajectory, -gripper_trajectory, 
                               interpolated_time, execute_time_factor=execute_time_factor)
    
    # Wait for completion
    trajectory_execution_time = max(interpolated_time) * execute_time_factor
    print(f"[{motion_name}] Waiting {trajectory_execution_time:.2f} seconds for completion...")
    rospy.sleep(trajectory_execution_time + 2.0)
    
    print(f"[{motion_name}] Motion completed successfully!")
    return True

if __name__ == "__main__":
    # Configuration
    urdf_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/urdf/open_manipulator_6dof.urdf'
    mesh_path = '/root/catkin_ws/src/open_manipulator_friends/open_manipulator_6dof_description/meshes'
    
    # Bag file paths
    pick_bag_path = '/root/catkin_ws/src/om_position_controller/recording/pick.bag'
    place_bag_path = '/root/catkin_ws/src/om_position_controller/recording/place.bag'  # Add your place bag path
    
    # DMP save paths
    pick_dmp_path = '/root/catkin_ws/src/om_position_controller/recording/pick_motion.pkl'
    place_dmp_path = '/root/catkin_ws/src/om_position_controller/recording/place_motion.pkl'
    
    # Home position
    home_position = [-0.03834952, -0.84062147, 1.26093221, 0.00613592, 1.97576725, -0.00460194]
    
    print("=== Starting Pick and Place Operation ===")
    
    try:
        # Initialize components
        dmp_gen = DMPMotionGenerator(
            urdf_path, 
            mesh_path,
            base_link="world"
        )
        
        publisher = GazeboTrajectoryPublisher()
        rospy.sleep(2.0)
        
        # 1. PICK MOTION - Blue Cube
        success = execute_motion(
            dmp_gen=dmp_gen,
            bag_path=pick_bag_path,
            dmp_save_path=pick_dmp_path,
            cube_name="blue_cube",
            position_offset=[0.0, 0.0, 0.01],  # Slight offset above cube
            publisher=publisher,
            motion_name="pick",
            execute_time_factor=5,
            visualize=False  # Set to False to skip visualization
        )
        
        if not success:
            print("Pick motion failed!")
            exit(1)
        
        # 2. RETURN TO HOME
        print("\n=== Returning to Home Position ===")
        publisher.publish_home_position(
            home_position=home_position,
            execution_time=5.0
        )
        print("[Home] Waiting for home position...")
        rospy.sleep(7.0)  # Wait for home position completion
        print("[Home] Home position reached!")
        
        # 3. PLACE MOTION - Green Cube
        success = execute_motion(
            dmp_gen=dmp_gen,
            bag_path=place_bag_path,
            dmp_save_path=place_dmp_path,
            cube_name="green_cube",
            position_offset=[0.0, 0.0, 0.07],  # Offset above green cube for placing
            publisher=publisher,
            motion_name="place",
            execute_time_factor=5,
            visualize=False  # Set to True if you want to visualize
        )
        
        if not success:
            print("Place motion failed!")
            exit(1)
        
        # 4. FINAL RETURN TO HOME
        print("\n=== Final Return to Home Position ===")
        publisher.publish_home_position(
            home_position=home_position,
            execution_time=5.0
        )
        print("[Final] Returning to home position...")
        rospy.sleep(7.0)
        
        print("\n=== Pick and Place Operation Completed Successfully! ===")
        
    except rospy.ROSInterruptException:
        print("[Main] ROS interrupted.")
    except Exception as e:
        print(f"[Main] Error during pick and place operation: {e}")
        import traceback
        traceback.print_exc()