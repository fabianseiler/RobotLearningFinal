#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from ultralytics import YOLO, SAM
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
import message_filters
from sklearn.decomposition import PCA
from tf.transformations import quaternion_about_axis
import tf
import time

class GPUObjectDetector:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('gpu_object_detector', anonymous=True)
        
        # Load parameters
        self.model_path = "best.pt" # TODO: change path to model
        self.confidence = 0.7 #rospy.get_param("~confidence", 0.5)
        self.sam_ckpt = "sam2_b.pt" #rospy.get_param("~sam_ckpt")
        self.object = ["one","two", "three", "four", "five"]    # TODO: change cube names
        self.prev_X = None
        self.prev_Y = None
        self.prev_Z = None
        self.threshold = 0.1  # Threshold for outlier rejection


        # Initialize models on GPU
        self.model = YOLO(self.model_path)
        self.sam_model = SAM(model=self.sam_ckpt)
        
        # Initialize CV bridge and tf broadcaster
        self.bridge = CvBridge()
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # Camera intrinsics
        self.K = None  # Camera matrix
        self.D = None  # Distortion coefficients
        
        # Get camera info once
        rospy.loginfo("Waiting for camera info...")
        camera_info = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
        self.setup_camera_info(camera_info)
        rospy.loginfo("Received camera info")
        rospy.loginfo("K: ")
        rospy.loginfo(self.K)
        rospy.loginfo("D: ")
        rospy.loginfo(self.D)
        
        # Subscribe to RGB and depth topics
        self.rgb_sub = message_filters.Subscriber('/camera/color/image_raw/compressed', CompressedImage)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        print("Subscribed to RGB and depth topics")
        
        # Time synchronizer
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        ts.registerCallback(self.callback)
        print("Setup finished")

    def setup_camera_info(self, camera_info):
        """Set up camera intrinsics from camera info message"""
        self.K = np.array(camera_info.K).reshape(3, 3)
        self.D = np.array(camera_info.D)
        
        # Store focal length and principal point
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

    def deproject_pixel_to_point(self, pixel_x, pixel_y, depth):
        """
        Convert pixel coordinates and depth to 3D point.
        
        Args:
            pixel_x (float): x coordinate in image
            pixel_y (float): y coordinate in image
            depth (float): depth value in meters
            
        Returns:
            tuple: (X, Y, Z) coordinates in camera frame
        """
        # Apply undistortion if needed (assuming points are already undistorted in this case)
        x = (pixel_x - self.cx) / self.fx
        y = (pixel_y - self.cy) / self.fy
        
        # Calculate 3D coordinates
        X = x * depth
        Y = y * depth
        Z = depth

        if self.prev_X is None:
            self.prev_X = X
            self.prev_Y = Y
            self.prev_Z = Z

        threshold = 0.2
        # Reject outliers based on a distance threshold
        # print(f"Prev: {self.prev_X}, {self.prev_Y}, {self.prev_Z}")
        # print(f"Curr: {X}, {Y}, {Z}")
        if np.linalg.norm(np.array([X - self.prev_X, Y - self.prev_Y, Z - self.prev_Z])) > threshold:
            rospy.logwarn("Pose jump detected, rejecting outlier")
            return (None, None, None)
        
        return (X, Y, Z)

    def callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and depth messages"""
        try:
            print("Received RGB and depth messages")
            # Convert compressed image to CV2 format
            color_image = self.bridge.compressed_imgmsg_to_cv2(rgb_msg)
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg)

            # # Apply a median filter to smooth the depth image
            # depth_image = cv2.medianBlur(depth_image, 5)

            # YOLO inference
            # s_det = time.process_time_ns()
            results = self.model.predict(source=color_image, conf=self.confidence, show=False, verbose=False)
            # print(f"Detected {len(results)} objects")
            boxes = results[0].boxes
            # print(f"Detected {len(boxes)} objects")
            # print(f"Object names: {self.model.names}")
            # e_det = time.process_time_ns()
            # try:
            #     print(len(boxes), self.model.names[int(boxes[0].cls[0])])
            # except:
            #     print(len(boxes))
            # print("Det time", (e_det - s_det)/10**9)
            # s_seg = time.process_time_ns()
            # Process each detection
            for idx, box in enumerate(boxes):
                class_idx = int(box.cls[0])
                class_name = self.model.names[class_idx]

                # print(class_name)

                if class_name not in self.object:
                    continue

                print(f"Detected {class_name} with confidence {box.conf[0]:.2f}")
                # Extract bounding box coordinates
                bbox = box.xyxy[0]
                x1, y1, x2, y2 = map(int, bbox.tolist())

                # Generate mask using SAM2
                sam_results = self.sam_model(color_image, bboxes=bbox.tolist(), verbose=False)
                mask = sam_results[0].masks
                mask = np.asarray(mask.xy)[0]

                # Process mask and compute orientation
                self.process_mask(mask, depth_image, x1, y1, x2, y2, idx, rgb_msg.header.stamp, rgb_msg.header.frame_id, class_name)
            e_seg = time.process_time_ns()
            # print("seg", (e_seg - s_seg)/10**9)
            
        except Exception as e:
            rospy.logerr(f"Error in callback: {str(e)}")

    def process_mask(self, mask, depth_image, x1, y1, x2, y2, idx, timestamp, frame_id, class_name):
        """Process mask to compute orientation and position"""
        try:
            # Perform PCA on mask coordinates
            pca = PCA(n_components=2)
            pca.fit(mask)
            components = pca.components_
            
            # Enforce consistent orientation for the second component
            # if components[1, 1] < 0:  # Check the y-component of the second principal axis
            #     components[1] = -components[1]  # Flip the second component if necessary

            
            # Compute orientation from principal component
            major_component = components[0]
            angle_rad = np.arctan2(major_component[1], major_component[0])

            
            quat = quaternion_about_axis(angle_rad, (0, 0, 1))

            # Compute 3D position
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Get depth value in meters (assuming depth is in millimeters)
            if (center_y - 10 >= 0 and center_y + 10 < depth_image.shape[0] and
                center_x - 10 >= 0 and center_x + 10 < depth_image.shape[1]):
                depth_array = depth_image[center_y-10:center_y+10, center_x-10:center_x+10]
                # print(f"Depth array: {depth_array}")
                print(f"Averaged depth: {np.average(depth_array[depth_array>0.])/1000.0}")
            else:
                rospy.logwarn("Selected region is out of depth image boundaries")
            depth = depth_image[center_y, center_x] / 1000.0
            print(f"Depth at center point: {depth} m")
            # Validate depth value
            # if depth <= 0 or np.isnan(depth):
            #     rospy.logwarn(f"Invalid depth value: {depth}")
            #     return

            #print(f"Center point: ({center_x}, {center_y}), Depth: {depth}")
            # Convert pixel coordinates to 3D point
            X, Y, Z = self.deproject_pixel_to_point(center_x, center_y, depth)

            # print(X, Y, Z)
            if X is None or Y is None or Z is None:
                rospy.logwarn("Invalid depth value, skipping broadcast")
                return

            # Example of exponential smoothing
            alpha = 0.5  # Smoothing factor
            
            
            self.prev_X = alpha * X + (1 - alpha) * self.prev_X
            self.prev_Y = alpha * Y + (1 - alpha) * self.prev_Y
            self.prev_Z = alpha * Z + (1 - alpha) * self.prev_Z
            
            # Broadcast transform if position is valid
            if not np.isnan(X) and not np.isnan(Y) and not np.isnan(Z):
                self.broadcast_tf(X, Y, Z, quat, timestamp, frame_id, class_name)

        except Exception as e:
            rospy.logerr(f"Error processing mask: {str(e)}")

    def broadcast_tf(self, x, y, z, quat, timestamp, frame_id, class_name):
        """Broadcast object pose as tf transform"""

        print(f"{class_name} position: x={x:.3f}, y={y:.3f}, z={z:.3f}")
        self.tf_broadcaster.sendTransform(
            (x, y, z),
            (quat[0], quat[1], quat[2], quat[3]),
            rospy.Time.now(),
            class_name,
            "camera_color_frame_calib"
        )

if __name__ == '__main__':
    try:
        detector = GPUObjectDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in TF Publisher: {e}")


 