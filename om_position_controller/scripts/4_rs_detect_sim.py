#!/usr/bin/env python3

import rospy
import tf2_ros
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped

class GazeboTFPublisher:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('gazebo_tf_publisher', anonymous=True)
        
        # Create TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Subscribe to gazebo model states
        self.model_states_sub = rospy.Subscriber(
            '/gazebo/model_states', 
            ModelStates, 
            self.model_states_callback
        )
        
        # Define which models to publish TF for (excluding ground_plane as it's typically the world frame)
        self.models_to_publish = ['green_cube', 'red_cube', 'blue_cube']
        
        # Track last published timestamps for each model to prevent duplicates
        self.last_published_times = {}
        
        # Minimum time difference between publications (in seconds)
        self.min_time_diff = 0.001  # 1ms minimum difference
        
        rospy.loginfo("Gazebo TF Publisher initialized")
        rospy.loginfo(f"Publishing TF for models: {self.models_to_publish}")
    
    def model_states_callback(self, msg):
        """
        Callback function that processes ModelStates messages and publishes TF transforms
        """
        transforms = []
        current_time = rospy.Time.now()
        
        # Check if enough time has passed since last publication
        if hasattr(self, 'last_callback_time'):
            time_diff = (current_time - self.last_callback_time).to_sec()
            if time_diff < self.min_time_diff:
                # Adjust current time to ensure it's unique
                current_time = self.last_callback_time + rospy.Duration(self.min_time_diff)
        
        self.last_callback_time = current_time
        
        # Process each model in the message
        for i, model_name in enumerate(msg.name):
            # Skip ground_plane and only publish TF for specified models
            if model_name in self.models_to_publish:
                # Check if we should publish for this model (avoid duplicate timestamps)
                if model_name in self.last_published_times:
                    time_since_last = (current_time - self.last_published_times[model_name]).to_sec()
                    if time_since_last < self.min_time_diff:
                        continue  # Skip this model to avoid duplicate timestamp
                
                # Create transform message
                transform = TransformStamped()
                
                # Set header
                transform.header.stamp = current_time
                transform.header.frame_id = "world"  # Parent frame (typically world or map)
                transform.child_frame_id = model_name
                
                # Set translation from position
                transform.transform.translation.x = msg.pose[i].position.x
                transform.transform.translation.y = msg.pose[i].position.y
                transform.transform.translation.z = msg.pose[i].position.z
                
                # Set rotation from orientation (quaternion)
                transform.transform.rotation.x = msg.pose[i].orientation.x
                transform.transform.rotation.y = msg.pose[i].orientation.y
                transform.transform.rotation.z = msg.pose[i].orientation.z
                transform.transform.rotation.w = msg.pose[i].orientation.w
                
                transforms.append(transform)
                
                # Update the last published time for this model
                self.last_published_times[model_name] = current_time
        
        # Broadcast all transforms
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)
    
    def run(self):
        """
        Main loop to keep the node running
        """
        rospy.loginfo("Starting TF publishing from Gazebo model states...")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down Gazebo TF Publisher")

def main():
    try:
        # Create and run the publisher
        tf_publisher = GazeboTFPublisher()
        tf_publisher.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in Gazebo TF Publisher: {e}")

if __name__ == '__main__':
    main()