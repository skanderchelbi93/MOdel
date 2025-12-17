#!/usr/bin/env python3
#
# Minimal FoundationPose launch file that uses:
#   - Your own /segmentation topic (from your Python code)
#   - Your own RGB + Depth topics
#   - FoundationPose TRT engines
#   - NO YOLO pipeline at all
#
# This replaces the entire YOLO → mask stage, but keeps FoundationPose unchanged.

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


# Default locations for TRT engines (same as NVIDIA)
REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'


def generate_launch_description():

    # --- Launch arguments you can pass from CLI ---
    mesh_file_path = LaunchConfiguration('mesh_file_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')

    launch_args = [
        DeclareLaunchArgument(
            'mesh_file_path',
            default_value='',
            description='Absolute path to the object mesh (.obj or .ply)'
        ),
        DeclareLaunchArgument(
            'refine_engine_file_path',
            default_value=REFINE_ENGINE_PATH,
            description='Path to the refine TensorRT engine'
        ),
        DeclareLaunchArgument(
            'score_engine_file_path',
            default_value=SCORE_ENGINE_PATH,
            description='Path to the score TensorRT engine'
        )
    ]

    # --- FoundationPose Node ---
    foundationpose_node = ComposableNode(
        name='foundationpose_node',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,

            # Refine engine
            'refine_engine_file_path': refine_engine_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],

            # Score engine
            'score_engine_file_path': score_engine_file_path,
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
        }],
        remappings=[
            # Your RGB + Depth topics (ensure your Python publishes the same!)
            ('pose_estimation/image',        'rgb/image_rect_color'),
            ('pose_estimation/depth_image',  'depth_image'),
            ('pose_estimation/camera_info',  'rgb/camera_info'),

            # 👉 Your custom segmentation topic
            ('pose_estimation/segmentation', 'segmentation'),

            # Output pose
            ('pose_estimation/output', 'output')
        ]
    )

    # --- Container for FoundationPose ---
    container = ComposableNodeContainer(
        name='foundationpose_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[foundationpose_node],
        output='screen'
    )

    return LaunchDescription(launch_args + [container])
