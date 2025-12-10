---
title: Chapter 3 - Practical ROS 2 Examples and Message Flow
sidebar_label: Practical Examples
---

# Chapter 3: Practical ROS 2 Examples and Message Flow

This chapter provides practical examples of ROS 2 message flow, helping you understand how different components communicate in real-world scenarios. Practical examples reinforce theoretical knowledge and help visualize how components work together in humanoid robot systems.

## Learning Objectives

After completing this chapter, you will be able to:
- Implement message flow examples in a ROS 2 system
- Observe messages passing between nodes in a ROS 2 system
- Trace message flow and understand communication patterns
- Apply message flow concepts to humanoid robot scenarios

## Example 1: Simple Publisher-Subscriber Pattern

Let's start with a simple example that demonstrates the basic publisher-subscriber pattern:

### Publisher Node: sensor_publisher.py

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(LaserScan, 'laser_scan', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_frame'

        # Simulate laser scan data
        msg.angle_min = -1.57  # -90 degrees
        msg.angle_max = 1.57   # 90 degrees
        msg.angle_increment = 0.1
        msg.time_increment = 0.0
        msg.scan_time = 0.0
        msg.range_min = 0.0
        msg.range_max = 10.0

        # Generate random range data
        ranges = [random.uniform(0.5, 5.0) for _ in range(32)]
        msg.ranges = ranges

        self.publisher.publish(msg)
        self.get_logger().info(f'Published laser scan with {len(msg.ranges)} readings')

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Node: obstacle_detector.py

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        self.subscription = self.create_subscription(
            LaserScan,
            'laser_scan',
            self.laser_callback,
            10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.subscription  # prevent unused variable warning

    def laser_callback(self, msg):
        # Check for obstacles in front (within 1 meter)
        front_distances = msg.ranges[len(msg.ranges)//2-5:len(msg.ranges)//2+5]
        min_distance = min(front_distances) if front_distances else float('inf')

        cmd = Twist()
        if min_distance < 1.0:
            # Obstacle detected, stop and turn
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn right
            self.get_logger().info(f'Obstacle detected! Distance: {min_distance:.2f}m, turning right')
        else:
            # Path clear, move forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            self.get_logger().info(f'Path clear! Distance: {min_distance:.2f}m, moving forward')

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    obstacle_detector = ObstacleDetector()

    try:
        rclpy.spin(obstacle_detector)
    except KeyboardInterrupt:
        pass
    finally:
        obstacle_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Message Flow Analysis

```
[Sensor Publisher] --(laser_scan msg)--> [Obstacle Detector] --(cmd_vel msg)--> [Robot Controller]
```

1. **Sensor Publisher** generates simulated laser scan data every 0.5 seconds
2. **Obstacle Detector** receives the laser scan data and processes it to detect obstacles
3. **Obstacle Detector** publishes velocity commands based on obstacle detection
4. **Robot Controller** (not shown) would receive the velocity commands and control the robot

## Example 2: Service-Based Communication

Let's look at a service-based example for requesting specific robot actions:

### Service Server: navigation_service.py

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from example_interfaces.srv import Trigger

class NavigationService(Node):
    def __init__(self):
        super().__init__('navigation_service')
        self.srv = self.create_service(
            Trigger,
            'navigate_to_goal',
            self.navigate_callback
        )
        self.current_goal = None
        self.get_logger().info('Navigation service ready')

    def navigate_callback(self, request, response):
        self.get_logger().info('Received navigation request')

        # Simulate navigation process
        # In a real system, this would interface with navigation stack
        success = True  # Simulated success

        if success:
            response.success = True
            response.message = 'Navigation completed successfully'
            self.get_logger().info('Navigation completed successfully')
        else:
            response.success = False
            response.message = 'Navigation failed'
            self.get_logger().info('Navigation failed')

        return response

def main(args=None):
    rclpy.init(args=args)
    navigation_service = NavigationService()

    try:
        rclpy.spin(navigation_service)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client: navigation_client.py

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger

class NavigationClient(Node):
    def __init__(self):
        super().__init__('navigation_client')
        self.cli = self.create_client(Trigger, 'navigate_to_goal')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Navigation service not available, waiting...')

        # Request navigation after a delay
        self.timer = self.create_timer(2.0, self.send_request)
        self.req = Trigger.Request()

    def send_request(self):
        self.get_logger().info('Sending navigation request...')
        self.future = self.cli.call_async(self.req)
        self.future.add_done_callback(self.response_callback)

    def response_callback(self, future):
        response = future.result()
        if response.success:
            self.get_logger().info(f'Navigation successful: {response.message}')
        else:
            self.get_logger().info(f'Navigation failed: {response.message}')

def main(args=None):
    rclpy.init(args=args)
    navigation_client = NavigationClient()

    try:
        rclpy.spin(navigation_client)
    except KeyboardInterrupt:
        pass
    finally:
        navigation_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Communication Flow

```
[Navigation Client] --(request)--> [Navigation Service] --(response)--> [Navigation Client]
```

## Example 3: Humanoid Robot Joint Control

Here's a more complex example for humanoid robot joint control:

### Joint Controller: joint_controller.py

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import math

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # Publisher for joint trajectory commands
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory',
            10
        )

        # Subscriber for current joint states
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for generating joint trajectories
        self.timer = self.create_timer(1.0, self.generate_trajectory)

        self.current_joints = None
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]

        self.get_logger().info('Joint controller initialized')

    def joint_state_callback(self, msg):
        self.current_joints = msg

    def generate_trajectory(self):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        # Create a trajectory point with sinusoidal motion
        point = JointTrajectoryPoint()

        # Generate joint positions based on time for walking pattern
        current_time = self.get_clock().now().nanoseconds / 1e9

        positions = []
        for i, joint_name in enumerate(self.joint_names):
            # Create different motion patterns for different joints
            if 'hip' in joint_name:
                pos = 0.1 * math.sin(current_time)
            elif 'knee' in joint_name:
                pos = 0.05 * math.sin(current_time + math.pi/2)
            else:  # ankle
                pos = 0.05 * math.sin(current_time + math.pi)
            positions.append(pos)

        point.positions = positions
        point.time_from_start.sec = 1
        point.time_from_start.nanosec = 0

        traj.points = [point]

        self.trajectory_pub.publish(traj)
        self.get_logger().info(f'Published joint trajectory for {len(positions)} joints')

def main(args=None):
    rclpy.init(args=args)
    joint_controller = JointController()

    try:
        rclpy.spin(joint_controller)
    except KeyboardInterrupt:
        pass
    finally:
        joint_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Example 4: Multi-Node Communication System

Let's create a complete system that demonstrates how multiple nodes work together:

### System Architecture

```
[Sensor Node] --(sensor_data)--> [AI Decision Node] --(commands)--> [Robot Controller]
      |                                    |
      v                                    v
[Localization] <----(pose)---- [Path Planner] ----(waypoints)----> [Navigation]
```

### AI Decision Node: ai_decision_maker.py

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Float32

class AIDecisionMaker(Node):
    def __init__(self):
        super().__init__('ai_decision_maker')

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(
            Pose, '/robot_pose', self.pose_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.risk_pub = self.create_publisher(Float32, '/risk_level', 10)

        self.current_pose = None
        self.get_logger().info('AI Decision Maker initialized')

    def scan_callback(self, msg):
        # Analyze sensor data and make decisions
        min_distance = min(msg.ranges) if msg.ranges else float('inf')

        # Calculate risk level based on proximity to obstacles
        risk_level = Float32()
        if min_distance < 0.5:
            risk_level.data = 0.9  # High risk
        elif min_distance < 1.0:
            risk_level.data = 0.5  # Medium risk
        else:
            risk_level.data = 0.1  # Low risk

        self.risk_pub.publish(risk_level)

        # Make navigation decision based on sensor data
        cmd = Twist()
        if min_distance < 0.7:
            # Too close to obstacle, turn
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        else:
            # Path clear, move forward
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

    def pose_callback(self, msg):
        self.current_pose = msg
        self.get_logger().info(f'Robot at position: ({msg.position.x:.2f}, {msg.position.y:.2f})')

def main(args=None):
    rclpy.init(args=args)
    ai_decision_maker = AIDecisionMaker()

    try:
        rclpy.spin(ai_decision_maker)
    except KeyboardInterrupt:
        pass
    finally:
        ai_decision_maker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Message Flow Visualization

Let's visualize the message flow with ASCII diagrams:

### Simple Navigation Flow:
```
    Sensor Data
         |
         v
    [Obstacle Detection] ----> [Movement Decision] ----> [Robot Movement]
         |                          |
         |                          |
         v                          v
    Obstacle Detected          Move Forward/Turn
```

### Complex System Flow:
```
    Environment Sensors
           |
           v
    [Perception Node] -----> [AI Decision Node] -----> [Control Node]
           |                       |                       |
           |                       |                       |
           v                       v                       v
    Object Detection        Risk Assessment         Robot Actuation
           |                       |                       |
           +----------+------------+                       |
                    |                                    |
                    v                                    v
            [Planning Node] <---------------------- [Robot State]
                    |
                    v
               Waypoint
               Generation
```

## Running the Examples

To run these examples:

1. **Terminal 1** - Start the publisher:
   ```bash
   ros2 run my_package sensor_publisher
   ```

2. **Terminal 2** - Start the subscriber:
   ```bash
   ros2 run my_package obstacle_detector
   ```

3. **Monitor messages**:
   ```bash
   ros2 topic echo /laser_scan
   ros2 topic echo /cmd_vel
   ```

## Key Takeaways

1. **Asynchronous Communication**: Topics enable decoupled, asynchronous communication between nodes
2. **Synchronous Communication**: Services provide request-response patterns for specific operations
3. **Message Types**: Different message types serve specific purposes (sensors, control, navigation)
4. **Real-time Considerations**: Timing and frequency of message publication affect system performance
5. **Error Handling**: Robust systems handle message loss and communication failures gracefully

## Summary

These practical examples demonstrate how ROS 2 message flow works in real-world scenarios. The publisher-subscriber pattern enables asynchronous communication, while services provide synchronous request-response interactions. Understanding these patterns is crucial for building effective humanoid robot systems that integrate sensors, AI decision-making, and control mechanisms.

In the next module, we'll explore the digital twin concept with Gazebo and Unity simulation environments.