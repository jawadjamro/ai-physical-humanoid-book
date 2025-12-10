---
title: Chapter 2 - VSLAM, Perception, and Path Planning with NVIDIA Isaac
sidebar_label: VSLAM and Perception
---

# Chapter 2: VSLAM, Perception, and Path Planning with NVIDIA Isaac

This chapter explores Visual Simultaneous Localization and Mapping (VSLAM), perception algorithms, and path planning systems using NVIDIA Isaac. These technologies form the core of the AI-robot brain, enabling humanoid robots to understand their environment, navigate safely, and plan optimal paths.

## Learning Objectives

After completing this chapter, you will be able to:
- Implement VSLAM systems for robot localization and mapping
- Develop perception algorithms for environment understanding
- Create path planning systems for robot navigation
- Integrate perception and navigation with NVIDIA Isaac tools
- Optimize perception and navigation for humanoid robot applications

## Visual Simultaneous Localization and Mapping (VSLAM)

VSLAM is a technique that allows robots to simultaneously localize themselves in an environment and build a map of that environment using visual sensors. For humanoid robots, VSLAM is crucial for autonomous navigation and spatial awareness.

### VSLAM Fundamentals

VSLAM combines computer vision and sensor fusion techniques to solve two problems simultaneously:
1. **Localization**: Where is the robot in the environment?
2. **Mapping**: What does the environment look like?

### Key Components of VSLAM

#### 1. Feature Detection and Matching
- **Feature Extraction**: Identify distinctive points in images (SIFT, ORB, FAST)
- **Feature Matching**: Match features between consecutive frames
- **Descriptor Computation**: Create unique descriptors for each feature

#### 2. Visual Odometry
- **Frame-to-Frame Motion**: Estimate motion between consecutive frames
- **Pose Estimation**: Calculate the robot's position and orientation
- **Drift Correction**: Minimize accumulated errors over time

#### 3. Loop Closure
- **Place Recognition**: Detect when the robot returns to a previously visited location
- **Map Optimization**: Correct accumulated drift using loop closure constraints
- **Global Consistency**: Maintain a consistent global map

### NVIDIA Isaac VSLAM Implementation

NVIDIA Isaac provides optimized VSLAM capabilities through Isaac ROS packages:

```yaml
# Example Isaac ROS VSLAM configuration
vslam_node:
  ros__parameters:
    # Camera parameters
    camera_matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    distortion_coefficients: [k1, k2, p1, p2, k3]

    # VSLAM parameters
    max_features: 1000
    min_feature_distance: 20
    tracking_threshold: 0.9
    relocalization_threshold: 0.5

    # Optimization settings
    bundle_adjustment_frequency: 10
    keyframe_selection_threshold: 0.1
```

### Example VSLAM Pipeline

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import cv2
import numpy as np

class IsaacVSLAMNode(Node):
    def __init__(self):
        super().__init__('isaac_vslam_node')

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_odom', 10)
        self.map_pub = self.create_publisher(Odometry, '/vslam_map', 10)

        # VSLAM components
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.previous_frame = None
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.keyframes = []

        self.camera_matrix = None
        self.distortion_coeffs = None

        self.get_logger().info('Isaac VSLAM node initialized')

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        image = self.ros_image_to_cv2(msg)

        if self.previous_frame is not None:
            # Extract features from current frame
            current_kp, current_desc = self.extract_features(image)

            # Match features with previous frame
            matches = self.match_features(
                self.previous_desc, current_desc)

            if len(matches) >= 10:  # Minimum matches for pose estimation
                # Estimate motion between frames
                motion = self.estimate_motion(
                    self.previous_kp, current_kp, matches)

                # Update current pose
                self.current_pose = self.current_pose @ motion

                # Publish odometry
                self.publish_odometry()

                # Check for keyframe
                if self.is_keyframe_needed():
                    self.add_keyframe(image, self.current_pose)

        # Store current frame for next iteration
        self.previous_frame = image.copy()
        self.previous_kp, self.previous_desc = self.extract_features(image)

    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp = self.feature_detector.detect(gray, None)
        kp, desc = self.feature_detector.compute(gray, kp)
        return kp, desc

    def match_features(self, desc1, desc2):
        if desc1 is not None and desc2 is not None:
            matches = self.descriptor_matcher.match(desc1, desc2)
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            return matches
        return []

    def estimate_motion(self, kp1, kp2, matches):
        if len(matches) >= 8:  # Minimum for fundamental matrix
            # Get matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                src_pts, dst_pts, self.camera_matrix,
                method=cv2.RANSAC, threshold=1.0)

            if E is not None:
                # Decompose essential matrix to get rotation and translation
                _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)

                # Create transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()

                return T

        return np.eye(4)  # Return identity if motion estimation fails

    def publish_odometry(self):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'

        # Convert transformation matrix to pose
        position = self.current_pose[:3, 3]
        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]

        # Convert rotation matrix to quaternion
        rotation = self.current_pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation)
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz

        self.odom_pub.publish(odom_msg)

    def rotation_matrix_to_quaternion(self, R):
        # Convert 3x3 rotation matrix to quaternion
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                qw = (R[2, 1] - R[1, 2]) / s
                qx = 0.25 * s
                qy = (R[0, 1] + R[1, 0]) / s
                qz = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                qw = (R[0, 2] - R[2, 0]) / s
                qx = (R[0, 1] + R[1, 0]) / s
                qy = 0.25 * s
                qz = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                qw = (R[1, 0] - R[0, 1]) / s
                qx = (R[0, 2] + R[2, 0]) / s
                qy = (R[1, 2] + R[2, 1]) / s
                qz = 0.25 * s

        return qw, qx, qy, qz

    def is_keyframe_needed(self):
        # Simple keyframe selection based on motion
        if len(self.keyframes) == 0:
            return True

        # Check if enough motion has occurred
        last_pose = self.keyframes[-1][1]
        motion = np.linalg.norm(self.current_pose[:3, 3] - last_pose[:3, 3])
        return motion > 0.5  # Keyframe every 0.5m

    def add_keyframe(self, image, pose):
        self.keyframes.append((image.copy(), pose.copy()))
        if len(self.keyframes) > 100:  # Limit keyframes to prevent memory issues
            self.keyframes.pop(0)

def main(args=None):
    rclpy.init(args=args)
    vslam_node = IsaacVSLAMNode()

    try:
        rclpy.spin(vslam_node)
    except KeyboardInterrupt:
        pass
    finally:
        vslam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Perception Systems with NVIDIA Isaac

### 1. Object Detection and Recognition

Isaac ROS provides optimized perception packages for object detection:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from isaac_ros_detectnet import DetectNetNode

class IsaacObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('isaac_object_detection')

        # Initialize Isaac DetectNet
        self.detectnet = DetectNetNode(
            input_topic_name='/camera/color/image_raw',
            output_topic_name='/detections',
            model_name='detectnet_coco',
            confidence_threshold=0.5
        )

        # Subscribe to camera data
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)

        # Publish detections
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/object_detections', 10)

    def image_callback(self, msg):
        # Process image through DetectNet
        detections = self.detectnet.process_image(msg)

        # Format and publish detections
        detection_array = Detection2DArray()
        detection_array.header = msg.header

        for detection in detections:
            detection_msg = Detection2D()
            detection_msg.header = msg.header
            detection_msg.bbox.center.x = detection.center_x
            detection_msg.bbox.center.y = detection.center_y
            detection_msg.bbox.size_x = detection.width
            detection_msg.bbox.size_y = detection.height

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection.class_name
            hypothesis.hypothesis.score = detection.confidence
            detection_msg.results.append(hypothesis)

            detection_array.detections.append(detection_msg)

        self.detection_pub.publish(detection_array)
```

### 2. Semantic Segmentation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from isaac_ros_segformer import SegformerNode

class IsaacSegmentationNode(Node):
    def __init__(self):
        super().__init__('isaac_segmentation')

        # Initialize Isaac Segformer for semantic segmentation
        self.segformer = SegformerNode(
            input_topic_name='/camera/color/image_raw',
            output_topic_name='/segmentation',
            model_name='segformer_ade20k'
        )

        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.segmentation_pub = self.create_publisher(Image, '/segmentation', 10)

    def image_callback(self, msg):
        # Process image for semantic segmentation
        segmentation_result = self.segformer.segment(msg)
        self.segmentation_pub.publish(segmentation_result)
```

### 3. Depth Estimation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from isaac_ros_midas import MidasNode

class IsaacDepthEstimationNode(Node):
    def __init__(self):
        super().__init__('isaac_depth_estimation')

        # Initialize Isaac MiDaS for depth estimation
        self.midas = MidasNode(
            input_topic_name='/camera/color/image_raw',
            output_topic_name='/depth_estimated',
            model_name='midas_v21_small'
        )

        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        self.depth_pub = self.create_publisher(Image, '/depth_estimated', 10)

    def image_callback(self, msg):
        # Estimate depth from monocular image
        depth_image = self.midas.estimate_depth(msg)
        self.depth_pub.publish(depth_image)
```

## Path Planning Systems

### 1. Global Path Planning (Navigation)

Global path planning creates an optimal path from start to goal considering the known map:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker
import numpy as np
import heapq

class IsaacGlobalPlannerNode(Node):
    def __init__(self):
        super().__init__('isaac_global_planner')

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        self.path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.marker_pub = self.create_publisher(Marker, '/path_visualization', 10)

        self.costmap = None
        self.map_resolution = 0.05  # 5cm per cell
        self.map_origin = None

    def map_callback(self, msg):
        self.costmap = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)

    def goal_callback(self, msg):
        if self.costmap is not None:
            # Convert goal pose to map coordinates
            goal_x = int((msg.pose.position.x - self.map_origin[0]) / self.map_resolution)
            goal_y = int((msg.pose.position.y - self.map_origin[1]) / self.map_resolution)

            # Get current robot position (simplified)
            current_x = int((-self.map_origin[0]) / self.map_resolution)  # Assuming robot at origin
            current_y = int((-self.map_origin[1]) / self.map_resolution)

            # Plan path using A* algorithm
            path = self.a_star_plan(current_x, current_y, goal_x, goal_y)

            if path:
                # Convert path back to world coordinates and publish
                world_path = self.convert_path_to_world(path)
                self.publish_path(world_path)

    def a_star_plan(self, start_x, start_y, goal_x, goal_y):
        """A* path planning algorithm"""
        # Check if start and goal are valid
        if (start_x < 0 or start_x >= self.costmap.shape[1] or
            start_y < 0 or start_y >= self.costmap.shape[0] or
            self.costmap[start_y, start_x] >= 50):  # Check if start is in obstacle
            return None

        if (goal_x < 0 or goal_x >= self.costmap.shape[1] or
            goal_y < 0 or goal_y >= self.costmap.shape[0] or
            self.costmap[goal_y, goal_x] >= 50):  # Check if goal is in obstacle
            return None

        # Define movement directions (8-connected)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        direction_costs = [1.414, 1, 1.414, 1, 1, 1.414, 1, 1.414]  # Diagonal vs straight costs

        # Initialize A* algorithm
        open_set = [(0, (start_x, start_y))]  # (f_score, (x, y))
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self.heuristic(start_x, start_y, goal_x, goal_y)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == (goal_x, goal_y):
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append((start_x, start_y))
                path.reverse()
                return path

            for i, direction in enumerate(directions):
                neighbor = (current[0] + direction[0], current[1] + direction[1])

                # Check bounds
                if (neighbor[0] < 0 or neighbor[0] >= self.costmap.shape[1] or
                    neighbor[1] < 0 or neighbor[1] >= self.costmap.shape[0]):
                    continue

                # Check if neighbor is in obstacle
                if self.costmap[neighbor[1], neighbor[0]] >= 50:  # 50 = obstacle threshold
                    continue

                # Calculate tentative g_score
                cost = direction_costs[i] + self.costmap[neighbor[1], neighbor[0]] / 100.0  # Normalize costmap value
                tentative_g_score = g_score[current] + cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor[0], neighbor[1], goal_x, goal_y)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, x1, y1, x2, y2):
        """Euclidean distance heuristic"""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def convert_path_to_world(self, path):
        """Convert path from map coordinates to world coordinates"""
        world_path = Path()
        world_path.header.frame_id = 'map'
        world_path.header.stamp = self.get_clock().now().to_msg()

        for x, y in path:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x * self.map_resolution + self.map_origin[0]
            pose.pose.position.y = y * self.map_resolution + self.map_origin[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation
            world_path.poses.append(pose)

        return world_path

    def publish_path(self, path):
        self.path_pub.publish(path)

def main(args=None):
    rclpy.init(args=args)
    planner_node = IsaacGlobalPlannerNode()

    try:
        rclpy.spin(planner_node)
    except KeyboardInterrupt:
        pass
    finally:
        planner_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Local Path Planning (Trajectory Generation)

Local path planning focuses on obstacle avoidance and dynamic trajectory generation:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
import numpy as np

class IsaacLocalPlannerNode(Node):
    def __init__(self):
        super().__init__('isaac_local_planner')

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.global_plan_sub = self.create_subscription(Path, '/global_plan', self.global_plan_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.current_pose = None
        self.laser_data = None
        self.global_plan = None
        self.local_goal = None

        # Robot parameters
        self.robot_radius = 0.3  # 30cm radius
        self.max_linear_speed = 0.5
        self.max_angular_speed = 1.0
        self.min_obstacle_distance = 0.5

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        self.laser_data = msg

    def global_plan_callback(self, msg):
        self.global_plan = msg.poses
        if self.global_plan:
            # Set next local goal from global plan
            self.local_goal = self.get_local_goal()

    def get_local_goal(self):
        """Get next goal from global plan that's within local horizon"""
        if not self.global_plan or not self.current_pose:
            return None

        current_pos = np.array([self.current_pose.position.x, self.current_pose.position.y])

        for pose in self.global_plan:
            goal_pos = np.array([pose.pose.position.x, pose.pose.position.y])
            distance = np.linalg.norm(goal_pos - current_pos)

            if distance > 1.0 and distance < 3.0:  # Local horizon: 1-3m
                return pose.pose
                break

        # If no suitable goal found, return the last one
        if self.global_plan:
            last_pose = self.global_plan[-1]
            return last_pose.pose

        return None

    def compute_velocity_command(self):
        """Compute velocity command based on local goal and obstacles"""
        if not self.current_pose or not self.local_goal or not self.laser_data:
            return None

        cmd = Twist()

        # Calculate direction to local goal
        current_pos = np.array([self.current_pose.position.x, self.current_pose.position.y])
        goal_pos = np.array([self.local_goal.position.x, self.local_goal.position.y])

        direction_to_goal = goal_pos - current_pos
        distance_to_goal = np.linalg.norm(direction_to_goal)

        if distance_to_goal < 0.2:  # Close enough to goal
            return cmd  # Stop

        # Normalize direction
        direction_to_goal = direction_to_goal / distance_to_goal

        # Check for obstacles in the direction of movement
        if self.laser_data:
            min_distance = min(self.laser_data.ranges)

            if min_distance < self.min_obstacle_distance:
                # Obstacle detected, avoid
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn right to avoid
                return cmd

        # Move towards goal
        cmd.linear.x = min(self.max_linear_speed, distance_to_goal * 0.5)  # Proportional to distance
        cmd.angular.z = np.arctan2(direction_to_goal[1], direction_to_goal[0]) * 0.5  # Proportional to angle

        # Limit angular velocity
        cmd.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, cmd.angular.z))

        return cmd

    def timer_callback(self):
        cmd = self.compute_velocity_command()
        if cmd:
            self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    local_planner = IsaacLocalPlannerNode()

    # Create timer for control loop
    local_planner.timer = local_planner.create_timer(0.1, local_planner.timer_callback)

    try:
        rclpy.spin(local_planner)
    except KeyboardInterrupt:
        pass
    finally:
        local_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Navigation Stack Integration

NVIDIA Isaac provides a complete navigation stack optimized for robotics:

```yaml
# Isaac Navigation configuration
isaac_navigation:
  ros__parameters:
    # Global planner parameters
    global_planner:
      planner_frequency: 1.0
      tolerance: 0.5
      use_dijkstra: true
      use_grid_path: false

    # Local planner parameters
    local_planner:
      controller_frequency: 20.0
      max_vel_x: 0.5
      min_vel_x: 0.1
      max_vel_theta: 1.0
      min_vel_theta: 0.1
      min_in_place_vel_theta: 0.4
      escape_vel: -0.1
      acc_lim_x: 2.5
      acc_lim_theta: 3.2

    # Costmap parameters
    global_costmap:
      update_frequency: 1.0
      publish_frequency: 1.0
      resolution: 0.05
      origin_x: 0.0
      origin_y: 0.0
      width: 40
      height: 40
      static_map: true

    local_costmap:
      update_frequency: 5.0
      publish_frequency: 2.0
      resolution: 0.05
      origin_x: 0.0
      origin_y: 0.0
      width: 10
      height: 10
      static_map: false
      rolling_window: true
```

## Performance Optimization

### 1. GPU Acceleration

Leverage NVIDIA GPUs for accelerated processing:

```python
import torch
import cv2

class IsaacPerceptionAccelerator:
    def __init__(self):
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load models to GPU
        self.detection_model = self.load_detection_model().to(self.device)
        self.segmentation_model = self.load_segmentation_model().to(self.device)

    def process_frame_gpu(self, image):
        # Convert image to tensor and move to GPU
        image_tensor = torch.from_numpy(image).to(self.device).float()

        # Process with GPU-accelerated models
        with torch.no_grad():
            detections = self.detection_model(image_tensor)
            segmentation = self.segmentation_model(image_tensor)

        return detections, segmentation
```

### 2. Multi-threading for Real-time Performance

```python
import threading
import queue

class IsaacRealtimePipeline:
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def process_loop(self):
        while True:
            try:
                image = self.input_queue.get(timeout=1)
                # Process image with perception pipeline
                result = self.perception_pipeline(image)
                self.output_queue.put(result)
            except queue.Empty:
                continue
```

## Best Practices for Humanoid Robot Navigation

### 1. Stability Considerations
- Account for robot dynamics and balance constraints
- Plan paths that consider step locations for bipedal robots
- Include balance recovery behaviors

### 2. Sensor Fusion
- Combine multiple sensor modalities (vision, LIDAR, IMU)
- Use Kalman filters or particle filters for state estimation
- Handle sensor failures gracefully

### 3. Computational Efficiency
- Optimize algorithms for real-time performance
- Use hierarchical planning (global coarse, local fine)
- Implement efficient data structures

## Summary

This chapter covered essential perception and navigation systems for humanoid robots using NVIDIA Isaac. VSLAM enables robots to understand their position and environment, while perception systems provide scene understanding through object detection, segmentation, and depth estimation. Path planning algorithms create safe and efficient navigation routes, combining global planning with local obstacle avoidance.

The integration of these systems with NVIDIA Isaac's optimized libraries and GPU acceleration enables real-time performance crucial for humanoid robot applications. Proper implementation of these technologies forms the AI-robot brain that enables autonomous behavior in complex environments.

In the next module, we'll explore Vision-Language-Action (VLA) systems that combine perception with cognitive planning and action execution.