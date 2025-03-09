from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('object_tracker'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='object_tracker',
            executable='tracker.py',
            name='tracker_node',
            output='screen',
            parameters=[config_path]
        )
    ])
