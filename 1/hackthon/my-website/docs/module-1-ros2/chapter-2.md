---
title: Chapter 2 - Python Agent to ROS 2 Communication
sidebar_label: Python Communication
---

# Chapter 2: Python Agent to ROS 2 Communication

This chapter demonstrates how Python-based Agents communicate with ROS 2 controllers using the rclpy library. This is essential for connecting AI systems (like Python-based agents) with physical or simulated robots running ROS 2.

## Learning Objectives

After completing this chapter, you will be able to:
- Use rclpy to create Python nodes that communicate with ROS 2 systems
- Publish messages to appropriate topics from Python agents
- Call services from Python agents to interact with ROS 2 controllers
- Process commands received by ROS 2 systems from Python agents

## Introduction to rclpy

**rclpy** is the Python client library for ROS 2. It allows Python programs to interact with ROS 2 systems, enabling Python-based AI agents to communicate with ROS 2-controlled robots. This library provides the necessary interfaces to create nodes, publish/subscribe to topics, and provide/call services.

### Key Features of rclpy

- **Node Creation**: Create and manage ROS 2 nodes in Python
- **Topic Communication**: Publish and subscribe to messages
- **Service Communication**: Provide and call services
- **Parameter Management**: Access and modify node parameters
- **Timer Support**: Execute callbacks at specific intervals
- **Action Support**: Implement and use action-based communication

## Setting Up rclpy

Before using rclpy, you need to initialize the ROS 2 Python client library:

```python
import rclpy
from rclpy.node import Node

def main(args=None):
    rclpy.init(args=args)  # Initialize ROS 2 client library
    # Create and run your node
    rclpy.shutdown()  # Shutdown when done
```

## Creating a Python Node

Let's create a simple Python node that can communicate with ROS 2:

```python
import rclpy
from rclpy.node import Node

class PythonAgentNode(Node):
    def __init__(self):
        super().__init__('python_agent_node')
        self.get_logger().info('Python Agent Node initialized')

def main(args=None):
    rclpy.init(args=args)
    agent_node = PythonAgentNode()

    try:
        rclpy.spin(agent_node)
    except KeyboardInterrupt:
        pass
    finally:
        agent_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publishing Messages to ROS 2

To send commands from a Python agent to a ROS 2 system, you'll need to publish messages to appropriate topics. Here's how to create a publisher:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CommandPublisher(Node):
    def __init__(self):
        super().__init__('command_publisher')
        self.publisher = self.create_publisher(String, 'robot_commands', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello Robot: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    command_publisher = CommandPublisher()

    try:
        rclpy.spin(command_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        command_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Subscribing to ROS 2 Topics

To receive data from a ROS 2 system, you'll need to create a subscriber:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')
        self.subscription = self.create_subscription(
            String,
            'robot_feedback',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    data_subscriber = DataSubscriber()

    try:
        rclpy.spin(data_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        data_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Calling ROS 2 Services

To request specific actions from a ROS 2 system, you can call services:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from example_interfaces.srv import AddTwoInts

class ServiceClient(Node):
    def __init__(self):
        super().__init__('service_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    service_client = ServiceClient()

    response = service_client.send_request(2, 3)
    service_client.get_logger().info(f'Result: {response.sum}')

    service_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Practical Example: AI Agent Controlling a Robot

Let's look at a practical example where a Python-based AI agent communicates with a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np

class AIController(Node):
    def __init__(self):
        super().__init__('ai_controller')

        # Publisher for robot movement commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for sensor data
        self.sensor_subscriber = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for AI decision making
        self.timer = self.create_timer(0.1, self.ai_decision_callback)

        self.current_joint_states = None
        self.get_logger().info('AI Controller initialized')

    def joint_state_callback(self, msg):
        self.current_joint_states = msg
        self.get_logger().info(f'Received joint states: {len(msg.name)} joints')

    def ai_decision_callback(self):
        if self.current_joint_states is not None:
            # Simple AI decision: move forward if joints are in safe range
            cmd = Twist()

            # Example: if first joint angle is less than 0.5, move forward
            if len(self.current_joint_states.position) > 0:
                if self.current_joint_states.position[0] < 0.5:
                    cmd.linear.x = 0.5  # Move forward
                else:
                    cmd.linear.x = 0.0  # Stop

            self.cmd_vel_publisher.publish(cmd)
            self.get_logger().info(f'Sent command: linear.x={cmd.linear.x}')

def main(args=None):
    rclpy.init(args=args)
    ai_controller = AIController()

    try:
        rclpy.spin(ai_controller)
    except KeyboardInterrupt:
        pass
    finally:
        ai_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with AI Frameworks

Python-based AI agents can integrate with popular AI frameworks like TensorFlow, PyTorch, or OpenAI libraries to make intelligent decisions and then communicate those decisions to ROS 2 systems:

```python
import rclpy
from rclpy.node import Node
import tensorflow as tf  # Example AI framework
from geometry_msgs.msg import Twist

class AIIntelligentController(Node):
    def __init__(self):
        super().__init__('ai_intelligent_controller')
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Load pre-trained AI model
        # self.ai_model = tf.keras.models.load_model('path/to/model')

        self.timer = self.create_timer(0.5, self.ai_processing_callback)

    def ai_processing_callback(self):
        # Process sensor data with AI model
        # decision = self.ai_model.predict(sensor_data)

        # Convert AI decision to ROS 2 command
        cmd = Twist()
        # cmd.linear.x = decision.linear_velocity
        # cmd.angular.z = decision.angular_velocity

        self.cmd_publisher.publish(cmd)
```

## Best Practices

1. **Error Handling**: Always include proper error handling for ROS 2 communications
2. **Resource Management**: Properly destroy nodes and clean up resources
3. **Threading**: Be aware of threading implications when using rclpy
4. **Message Types**: Use appropriate message types for your specific use case
5. **Logging**: Use ROS 2's logging system for debugging and monitoring

## Summary

Python agents can effectively communicate with ROS 2 systems using the rclpy library. This enables AI decision-making to be integrated with robot control, allowing for sophisticated autonomous behaviors. The combination of Python's AI capabilities with ROS 2's robotics infrastructure provides a powerful platform for developing intelligent robotic systems.

In the next chapter, we'll explore practical examples of ROS 2 message flow to reinforce these concepts.