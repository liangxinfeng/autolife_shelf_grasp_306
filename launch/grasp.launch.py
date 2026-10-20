from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([Node(package='autolife_shelf_grasp_306', executable='shelf_grasp', output='screen')])
