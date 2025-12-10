---
title: Chapter 1 - Understanding ROS 2 Core Concepts
sidebar_label: Core Concepts
---

# Chapter 1: Understanding ROS 2 Core Concepts

This chapter introduces the fundamental concepts of ROS 2 (Nodes, Topics, Services, Parameters) that form the foundation for controlling humanoid robots. Understanding these concepts is essential before diving into more complex ROS 2 interactions and Python agent integration.

## Learning Objectives

After completing this chapter, you will be able to:
- Identify and explain the purpose of Nodes, Topics, Services, and Parameters in a ROS 2 system
- Distinguish between publisher/subscriber and client/service patterns
- Understand how these components work together in a ROS 2 system

## What is ROS 2?

ROS 2 (Robot Operating System 2) is not an operating system but rather a flexible framework for writing robot software. It provides a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Key Features of ROS 2

- **Distributed computing**: Multiple processes can communicate seamlessly across different machines
- **Language independence**: Support for multiple programming languages (C++, Python, Rust, etc.)
- **Real-time support**: Improved real-time capabilities compared to ROS 1
- **Security**: Built-in security features for safe robot operation
- **Cross-platform**: Works on various operating systems and hardware platforms

## Core Concepts

### Nodes

A **Node** is the fundamental unit of computation in ROS 2. It's a process that performs computation and communicates with other nodes through messages. Nodes are organized into packages, which contain source code, data, and configuration files.

Key characteristics of nodes:
- Each node runs a specific task or function
- Nodes can be written in different programming languages
- Nodes communicate with each other through topics, services, or parameters
- Nodes can be launched individually or as part of a larger system

Example of a simple ROS 2 node structure:
```python
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        # Node initialization code here
```

### Topics

A **Topic** is a named bus over which nodes exchange messages in a publisher-subscriber pattern. This enables asynchronous communication between nodes where publishers send data and subscribers receive data without direct connection.

Key characteristics of topics:
- Data flows from publishers to subscribers
- Multiple publishers can publish to the same topic
- Multiple subscribers can subscribe to the same topic
- Communication is asynchronous and unidirectional
- Topics use a publish-subscribe communication model

### Services

A **Service** is a request-response communication pattern between nodes. Unlike topics, services provide synchronous communication where a client sends a request and waits for a response from a server.

Key characteristics of services:
- Synchronous communication model
- Request-response pattern
- One client communicates with one server at a time
- Used for operations that require a response or confirmation
- Blocking until response is received

### Parameters

**Parameters** are configuration values that can be set at runtime and are accessible to nodes. They provide a way to configure node behavior without recompiling code.

Key characteristics of parameters:
- Dynamic configuration at runtime
- Key-value pairs with various data types
- Can be set via launch files, command line, or programmatically
- Persist across node restarts when properly configured

## Communication Patterns

### Publisher-Subscriber Pattern

The publisher-subscriber pattern is the most common communication method in ROS 2. Publishers send messages to a topic, and any number of subscribers can receive those messages.

```
[Publisher Node] -----> [Topic] -----> [Subscriber Node]
                    (asynchronous)
```

### Client-Service Pattern

The client-service pattern provides synchronous request-response communication.

```
[Client Node] <-----> [Service Server]
    (synchronous)
```

## Practical Example

Let's look at how these concepts work together in a simple humanoid robot scenario:

1. **Sensor Node**: Publishes sensor data (e.g., camera images, IMU readings) to topics
2. **Controller Node**: Subscribes to sensor data and publishes motor commands to topics
3. **Navigation Service**: Provides path planning as a service that other nodes can request
4. **Configuration Parameters**: Store robot-specific values like joint limits, safety thresholds

## Summary

Understanding these core concepts is crucial for working with ROS 2 and humanoid robots. Each concept plays a specific role in the overall system architecture, and they work together to create a flexible and powerful robotics framework.

In the next chapter, we'll explore how Python-based agents communicate with ROS 2 controllers using the rclpy library.