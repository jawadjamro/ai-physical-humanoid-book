---
title: Chapter 2 - Cognitive Planning with LLMs and ROS 2 Action Sequencing
sidebar_label: Cognitive Planning
---

# Chapter 2: Cognitive Planning with LLMs and ROS 2 Action Sequencing

This chapter explores how Large Language Models (LLMs) can be integrated with ROS 2 to create cognitive planning systems for humanoid robots. We'll cover the implementation of LLM-based reasoning, action sequencing, and the integration with ROS 2 action servers for robust execution.

## Learning Objectives

After completing this chapter, you will be able to:
- Implement cognitive planning using Large Language Models
- Design ROS 2 action servers for complex task execution
- Create action sequencing pipelines that combine LLM reasoning with ROS 2 execution
- Handle uncertainty and errors in LLM-based planning
- Integrate multimodal perception with cognitive planning

## Cognitive Planning Architecture

Cognitive planning in VLA systems involves creating a bridge between high-level natural language commands and low-level robot actions. The architecture typically includes:

### 1. Command Interpretation Layer
- **Natural Language Understanding**: Parse and understand user commands
- **Context Integration**: Incorporate environmental and historical context
- **Goal Decomposition**: Break complex commands into achievable subgoals

### 2. Reasoning Layer
- **Knowledge Integration**: Access to world knowledge and robot capabilities
- **Plan Generation**: Create step-by-step action plans
- **Constraint Checking**: Verify plan feasibility and safety

### 3. Execution Layer
- **Action Sequencing**: Convert high-level plans to ROS 2 actions
- **Monitoring**: Track execution progress and handle deviations
- **Adaptation**: Modify plans based on execution feedback

## Implementing LLM-Based Cognitive Planning

### 1. LLM Integration for Task Planning

```python
import openai
import json
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import String

class LLMBasedPlanner(Node):
    def __init__(self):
        super().__init__('llm_planner')

        # OpenAI client
        self.client = openai.OpenAI()

        # Publishers and subscribers
        self.plan_pub = self.create_publisher(String, '/generated_plan', 10)
        self.feedback_pub = self.create_publisher(String, '/planning_feedback', 10)

        # Robot capabilities database
        self.robot_capabilities = self.load_robot_capabilities()

    def load_robot_capabilities(self):
        """Load robot's available capabilities"""
        return {
            "navigation": {
                "actions": ["move_to", "explore", "return_to_base", "follow_path"],
                "constraints": {"max_speed": 0.5, "min_turn_radius": 0.3}
            },
            "manipulation": {
                "actions": ["pick", "place", "grasp", "release", "transport"],
                "constraints": {"max_payload": 2.0, "reach_distance": 1.0}
            },
            "perception": {
                "actions": ["detect_object", "recognize_person", "measure_distance", "scan_area"],
                "constraints": {"detection_range": 3.0, "accuracy": 0.95}
            },
            "communication": {
                "actions": ["speak", "listen", "display_message"],
                "constraints": {"max_speech_length": 1000}
            }
        }

    def generate_plan(self, user_command, environment_context=None):
        """Generate execution plan using LLM"""
        # Construct system prompt with robot capabilities
        system_prompt = f"""
        You are an AI planning assistant for a humanoid robot. The robot has these capabilities:

        {json.dumps(self.robot_capabilities, indent=2)}

        Your role is to:
        1. Understand the user's command
        2. Consider the environmental context if provided
        3. Generate a step-by-step plan using only the robot's available actions
        4. Include necessary parameters for each action
        5. Consider safety and feasibility constraints
        6. Provide error handling steps if needed

        Respond with a JSON object containing:
        {{
            "command": "original user command",
            "interpretation": "what the command means",
            "plan": [
                {{
                    "step": 1,
                    "action": "action_name",
                    "parameters": {{"param1": "value1", "param2": "value2"}},
                    "description": "what this step does",
                    "expected_outcome": "what should happen"
                }}
            ],
            "estimated_duration": "estimated time in seconds",
            "potential_risks": ["list of potential issues"]
        }}
        """

        # Construct user message
        user_message = f"User command: '{user_command}'"
        if environment_context:
            user_message += f"\nEnvironmental context: {environment_context}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            plan_data = json.loads(response.choices[0].message.content)
            return plan_data

        except Exception as e:
            self.get_logger().error(f'Error generating plan: {e}')
            return None

    def validate_plan(self, plan):
        """Validate plan feasibility"""
        for step in plan.get('plan', []):
            action = step.get('action')

            # Check if action is available
            action_available = False
            for capability_category in self.robot_capabilities.values():
                if action in capability_category['actions']:
                    action_available = True
                    break

            if not action_available:
                return False, f"Action '{action}' not available"

            # Check constraints for each action
            if action in self.robot_capabilities.get('manipulation', {}).get('actions', []):
                payload = step.get('parameters', {}).get('payload', 0)
                if payload > self.robot_capabilities['manipulation']['constraints']['max_payload']:
                    return False, f"Payload {payload}kg exceeds maximum {self.robot_capabilities['manipulation']['constraints']['max_payload']}kg"

        return True, "Plan is valid"

    def execute_plan(self, plan):
        """Execute the generated plan"""
        for step in plan.get('plan', []):
            self.get_logger().info(f'Executing step {step["step"]}: {step["description"]}')

            # Execute action based on type
            success = self.execute_action(step)

            if not success:
                self.get_logger().error(f'Step {step["step"]} failed')
                return False

        return True

    def execute_action(self, step):
        """Execute individual action"""
        action = step.get('action')
        parameters = step.get('parameters', {})

        # This is where you would call specific ROS 2 services/actions
        # For example:
        if action == 'move_to':
            return self.execute_navigation(parameters)
        elif action == 'pick':
            return self.execute_manipulation('pick', parameters)
        elif action == 'place':
            return self.execute_manipulation('place', parameters)
        elif action == 'detect_object':
            return self.execute_perception(parameters)
        elif action == 'speak':
            return self.execute_communication(parameters)

        return False  # Unknown action
```

### 2. ROS 2 Action Server Implementation

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from robot_task_msgs.action import ExecuteTask
from geometry_msgs.msg import Pose
from std_msgs.msg import String

class CognitiveActionServer(Node):
    def __init__(self):
        super().__init__('cognitive_action_server')

        # Create action server
        self._action_server = ActionServer(
            self,
            ExecuteTask,
            'execute_cognitive_task',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())

        # LLM planner integration
        self.llm_planner = LLMBasedPlanner()

        self.get_logger().info('Cognitive Action Server initialized')

    def goal_callback(self, goal_request):
        """Accept or reject goal requests"""
        self.get_logger().info(f'Received task: {goal_request.task_description}')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the goal"""
        self.get_logger().info('Executing goal...')

        feedback_msg = ExecuteTask.Feedback()
        result = ExecuteTask.Result()

        # Generate plan using LLM
        plan_data = self.llm_planner.generate_plan(
            goal_handle.request.task_description,
            goal_handle.request.environment_context
        )

        if not plan_data:
            result.success = False
            result.message = 'Failed to generate plan'
            goal_handle.succeed()
            return result

        # Validate plan
        is_valid, validation_msg = self.llm_planner.validate_plan(plan_data)
        if not is_valid:
            result.success = False
            result.message = f'Plan validation failed: {validation_msg}'
            goal_handle.succeed()
            return result

        # Execute plan step by step
        total_steps = len(plan_data.get('plan', []))
        completed_steps = 0

        for step in plan_data.get('plan', []):
            if goal_handle.is_cancel_requested:
                result.success = False
                result.message = 'Task cancelled'
                goal_handle.canceled()
                return result

            # Update feedback
            feedback_msg.current_step = step['step']
            feedback_msg.total_steps = total_steps
            feedback_msg.current_action = step['description']
            goal_handle.publish_feedback(feedback_msg)

            # Execute action
            success = self.llm_planner.execute_action(step)

            if not success:
                result.success = False
                result.message = f'Step {step["step"]} failed: {step["description"]}'
                goal_handle.succeed()
                return result

            completed_steps += 1
            self.get_logger().info(f'Step {completed_steps}/{total_steps} completed')

        # Task completed successfully
        result.success = True
        result.message = f'Task completed successfully in {completed_steps} steps'
        goal_handle.succeed()
        return result

def main(args=None):
    rclpy.init(args=args)

    cognitive_server = CognitiveActionServer()

    # Use multi-threaded executor to handle callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(cognitive_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cognitive_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3. Action Sequencing and Coordination

```python
import asyncio
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String

class ActionSequencer(Node):
    def __init__(self):
        super().__init__('action_sequencer')

        # Action clients for different capabilities
        self.navigation_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.manipulation_client = ActionClient(self, MoveGroup, 'move_group')
        self.perception_client = ActionClient(self, DetectObjects, 'detect_objects')

        # Publishers for coordination
        self.status_pub = self.create_publisher(String, '/action_status', 10)
        self.coordination_pub = self.create_publisher(String, '/coordination_signals', 10)

        # Task queue for sequencing
        self.task_queue = asyncio.Queue()
        self.current_task = None

        # Start task processing
        self.process_tasks_timer = self.create_timer(0.1, self.process_task_queue)

    async def process_task_queue(self):
        """Process tasks from the queue"""
        if not self.task_queue.empty() and self.current_task is None:
            self.current_task = await self.task_queue.get()
            await self.execute_task(self.current_task)
            self.current_task = None

    async def execute_task(self, task):
        """Execute a complex task with multiple coordinated actions"""
        task_id = task.get('id')
        steps = task.get('steps', [])

        self.get_logger().info(f'Starting task {task_id} with {len(steps)} steps')

        for i, step in enumerate(steps):
            if self.is_task_cancelled(task_id):
                break

            # Update task status
            status_msg = String()
            status_msg.data = f"Task {task_id}: Step {i+1}/{len(steps)} - {step['description']}"
            self.status_pub.publish(status_msg)

            # Execute step
            success = await self.execute_step(step)

            if not success:
                self.get_logger().error(f'Step {i+1} failed in task {task_id}')
                # Implement error recovery or abort
                break

        # Task completion
        completion_msg = String()
        completion_msg.data = f"Task {task_id} completed"
        self.status_pub.publish(completion_msg)

    async def execute_step(self, step):
        """Execute a single step with appropriate action client"""
        action_type = step.get('action_type')
        parameters = step.get('parameters', {})

        if action_type == 'navigation':
            return await self.execute_navigation_step(parameters)
        elif action_type == 'manipulation':
            return await self.execute_manipulation_step(parameters)
        elif action_type == 'perception':
            return await self.execute_perception_step(parameters)
        else:
            self.get_logger().error(f'Unknown action type: {action_type}')
            return False

    async def execute_navigation_step(self, params):
        """Execute navigation step"""
        # Wait for action server
        self.navigation_client.wait_for_server()

        # Create goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.create_pose_from_params(params)

        # Send goal and wait for result
        goal_handle = await self.navigation_client.send_goal_async(goal_msg)
        result = await goal_handle.get_result_async()

        return result.result.success if result.result else False

    async def execute_manipulation_step(self, params):
        """Execute manipulation step"""
        self.manipulation_client.wait_for_server()

        goal_msg = MoveGroup.Goal()
        # Configure manipulation goal based on parameters
        goal_msg.request = self.create_manipulation_request(params)

        goal_handle = await self.manipulation_client.send_goal_async(goal_msg)
        result = await goal_handle.get_result_async()

        return result.result.success if result.result else False

    async def execute_perception_step(self, params):
        """Execute perception step"""
        self.perception_client.wait_for_server()

        goal_msg = DetectObjects.Goal()
        # Configure perception goal based on parameters
        goal_msg.target_objects = params.get('target_objects', [])

        goal_handle = await self.perception_client.send_goal_async(goal_msg)
        result = await goal_handle.get_result_async()

        return result.result.found_objects if result.result else []
```

## Handling Uncertainty and Errors

### 1. Uncertainty Management

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class UncertaintyEstimate:
    mean: float
    variance: float
    confidence: float

class UncertaintyManager:
    def __init__(self):
        self.uncertainty_models = {}
        self.observation_history = {}

    def update_uncertainty(self, observation: Dict[str, Any], sensor_type: str):
        """Update uncertainty based on new observation"""
        if sensor_type not in self.observation_history:
            self.observation_history[sensor_type] = []

        self.observation_history[sensor_type].append(observation)

        # Calculate uncertainty based on observation consistency
        if len(self.observation_history[sensor_type]) > 1:
            recent_observations = self.observation_history[sensor_type][-5:]  # Last 5 observations
            values = [obs.get('value', 0) for obs in recent_observations]

            mean_val = np.mean(values)
            variance = np.var(values)
            confidence = 1.0 / (1.0 + variance)  # Lower variance = higher confidence

            return UncertaintyEstimate(mean_val, variance, confidence)

        return UncertaintyEstimate(observation.get('value', 0), 1.0, 0.5)

    def adjust_plan_for_uncertainty(self, plan: Dict[str, Any], uncertainty: UncertaintyEstimate):
        """Adjust plan based on uncertainty level"""
        if uncertainty.confidence < 0.3:  # High uncertainty
            # Add verification steps
            verification_step = {
                "step": len(plan.get('plan', [])) + 1,
                "action": "verify_condition",
                "parameters": {"timeout": 10.0},
                "description": "Verify that action was successful",
                "expected_outcome": "Confirmation of action success"
            }
            plan['plan'].append(verification_step)
        elif uncertainty.confidence < 0.7:  # Medium uncertainty
            # Add error handling
            plan['error_handling'] = {
                "retry_count": 3,
                "timeout": 30.0,
                "fallback_action": "abort_and_report"
            }

        return plan
```

### 2. Error Recovery Strategies

```python
class ErrorRecoveryManager:
    def __init__(self):
        self.recovery_strategies = {
            'navigation_failure': self.recovery_navigation_failure,
            'manipulation_failure': self.recovery_manipulation_failure,
            'perception_failure': self.recovery_perception_failure,
            'communication_failure': self.recovery_communication_failure
        }

    def handle_error(self, error_type: str, context: Dict[str, Any]):
        """Handle specific error types with appropriate recovery"""
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](context)
        else:
            return self.generic_recovery(error_type, context)

    def recovery_navigation_failure(self, context):
        """Recovery for navigation failures"""
        current_pose = context.get('current_pose')
        target_pose = context.get('target_pose')
        failure_reason = context.get('failure_reason', '')

        recovery_plan = []

        if 'obstacle' in failure_reason:
            # Try alternative path
            recovery_plan.append({
                "action": "find_alternative_path",
                "parameters": {"start": current_pose, "goal": target_pose},
                "description": "Find alternative navigation route"
            })
        elif 'localization' in failure_reason:
            # Re-localize
            recovery_plan.append({
                "action": "relocalize_robot",
                "parameters": {},
                "description": "Re-establish robot position"
            })

        return recovery_plan

    def recovery_manipulation_failure(self, context):
        """Recovery for manipulation failures"""
        object_info = context.get('object_info')
        failure_reason = context.get('failure_reason', '')

        recovery_plan = []

        if 'grasp_failure' in failure_reason:
            # Try different grasp approach
            recovery_plan.extend([
                {
                    "action": "reassess_object_grasp",
                    "parameters": {"object": object_info},
                    "description": "Analyze object for alternative grasp points"
                },
                {
                    "action": "attempt_alternative_grasp",
                    "parameters": {"approach": "side_grasp"},
                    "description": "Try side grasp instead of top grasp"
                }
            ])

        return recovery_plan

    def generic_recovery(self, error_type, context):
        """Generic recovery for unknown error types"""
        return [
            {
                "action": "report_error",
                "parameters": {"error_type": error_type, "context": context},
                "description": f"Report {error_type} for human intervention"
            },
            {
                "action": "return_to_safe_state",
                "parameters": {},
                "description": "Return robot to safe configuration"
            }
        ]
```

## Integration with ROS 2 Ecosystem

### 1. Behavior Trees for Complex Task Management

```python
# Using py_trees for behavior tree implementation
import py_trees
import py_trees_ros
from std_msgs.msg import Bool

class VLABehaviorTreeManager(Node):
    def __init__(self):
        super().__init__('vla_behavior_tree_manager')

        # Create behavior tree
        self.root = self.create_behavior_tree()

        # Setup tree manager
        self.tree_manager = py_trees_ros.trees.BehaviourTree(
            root=self.root,
            namespace='vla_bt'
        )

        # Publishers for monitoring
        self.status_pub = self.create_publisher(Bool, '/behavior_tree_status', 10)

    def create_behavior_tree(self):
        """Create behavior tree for VLA tasks"""
        # Main sequence
        root = py_trees.composites.Sequence(name="VLA_Task_Sequence")

        # Add task components
        root.add_child(self.create_perception_branch())
        root.add_child(self.create_reasoning_branch())
        root.add_child(self.create_action_branch())

        return root

    def create_perception_branch(self):
        """Create perception branch of the tree"""
        perception_sequence = py_trees.composites.Sequence(name="Perception")

        # Add perception behaviors
        perception_sequence.add_child(
            py_trees.behaviours.WaitForTopic(
                name="Wait for image",
                topic_name="/camera/image_raw",
                msg_type=sensor_msgs.msg.Image
            )
        )

        perception_sequence.add_child(
            py_trees.behaviours.WaitForTopic(
                name="Wait for point cloud",
                topic_name="/camera/depth/points",
                msg_type=sensor_msgs.msg.PointCloud2
            )
        )

        return perception_sequence

    def create_reasoning_branch(self):
        """Create reasoning branch of the tree"""
        reasoning_sequence = py_trees.composites.Sequence(name="Reasoning")

        # Add reasoning behaviors
        reasoning_sequence.add_child(
            py_trees_ros.actions.Action(
                name="LLM Planning",
                action_type="task_planning_msgs/PlanTask",
                action_name="plan_task",
                action_goal=self.prepare_llm_goal()
            )
        )

        return reasoning_sequence

    def create_action_branch(self):
        """Create action branch of the tree"""
        action_selector = py_trees.composites.Selector(name="Action_Selection")

        # Add action sequences
        navigation_sequence = py_trees.composites.Sequence(name="Navigation")
        navigation_sequence.add_child(self.create_navigation_task())
        navigation_sequence.add_child(self.create_reaching_task())

        manipulation_sequence = py_trees.composites.Sequence(name="Manipulation")
        manipulation_sequence.add_child(self.create_approach_task())
        manipulation_sequence.add_child(self.create_grasp_task())

        action_selector.add_child(navigation_sequence)
        action_selector.add_child(manipulation_sequence)

        return action_selector
```

### 2. Safety and Monitoring Integration

```python
import threading
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy

class VLAMonitoringNode(Node):
    def __init__(self):
        super().__init__('vla_monitoring')

        # Safety monitoring
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Safety parameters
        self.safety_limits = {
            'joint_position': {'min': -3.14, 'max': 3.14},
            'joint_velocity': {'max': 2.0},
            'linear_velocity': {'max': 0.5},
            'angular_velocity': {'max': 1.0}
        }

        # Safety publisher
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)

        # Monitoring timer
        self.monitoring_timer = self.create_timer(0.1, self.safety_monitor)

        # Joint state cache
        self.current_joint_states = JointState()

    def joint_state_callback(self, msg):
        """Monitor joint states for safety"""
        self.current_joint_states = msg
        self.check_joint_limits()

    def cmd_vel_callback(self, msg):
        """Monitor velocity commands for safety"""
        linear_speed = abs(msg.linear.x)
        angular_speed = abs(msg.angular.z)

        if (linear_speed > self.safety_limits['linear_velocity']['max'] or
            angular_speed > self.safety_limits['angular_velocity']['max']):
            self.trigger_safety_stop("Velocity limits exceeded")

    def check_joint_limits(self):
        """Check if joint states are within safe limits"""
        for i, position in enumerate(self.current_joint_states.position):
            if (position < self.safety_limits['joint_position']['min'] or
                position > self.safety_limits['joint_position']['max']):
                self.trigger_safety_stop(f"Joint {i} position limit exceeded")

        for i, velocity in enumerate(self.current_joint_states.velocity):
            if abs(velocity) > self.safety_limits['joint_velocity']['max']:
                self.trigger_safety_stop(f"Joint {i} velocity limit exceeded")

    def safety_monitor(self):
        """Continuous safety monitoring"""
        # Additional safety checks can be added here
        pass

    def trigger_safety_stop(self, reason):
        """Trigger emergency stop"""
        self.get_logger().error(f"Safety violation: {reason}")

        # Publish emergency stop
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)
```

## Best Practices for Cognitive Planning

### 1. Plan Validation and Verification

```python
class PlanValidator:
    def __init__(self, robot_model):
        self.robot_model = robot_model

    def validate_plan(self, plan):
        """Validate plan for feasibility and safety"""
        validation_results = {
            'feasible': True,
            'safety_violations': [],
            'constraint_violations': [],
            'warnings': []
        }

        # Check each step in the plan
        for step in plan.get('plan', []):
            action = step.get('action')
            params = step.get('parameters', {})

            # Validate action parameters
            if not self.validate_action_parameters(action, params):
                validation_results['constraint_violations'].append(
                    f"Invalid parameters for action {action}"
                )
                validation_results['feasible'] = False

            # Check for safety constraints
            if not self.check_safety_constraints(action, params):
                validation_results['safety_violations'].append(
                    f"Safety constraint violation in action {action}"
                )
                validation_results['feasible'] = False

        return validation_results

    def validate_action_parameters(self, action, params):
        """Validate action-specific parameters"""
        # Implementation for parameter validation
        return True

    def check_safety_constraints(self, action, params):
        """Check safety constraints for action"""
        # Implementation for safety constraint checking
        return True
```

### 2. Performance Optimization

- **Caching**: Cache frequently used plans and knowledge
- **Parallel Processing**: Execute independent actions in parallel
- **Preemptive Planning**: Plan next steps while executing current ones
- **Resource Management**: Optimize computational resource usage

## Summary

This chapter covered the implementation of cognitive planning systems using Large Language Models integrated with ROS 2. We explored how to create LLM-based planners that can interpret natural language commands, generate executable action plans, and coordinate with ROS 2 action servers for robust execution.

The integration of LLMs with ROS 2 enables humanoid robots to perform complex tasks by breaking down high-level commands into sequences of executable actions. Proper error handling, uncertainty management, and safety monitoring are crucial for reliable operation in real-world environments.

In the next module, we'll explore the capstone project: implementing an autonomous humanoid robot that receives voice commands, plans actions, navigates, perceives objects, and manipulates them.