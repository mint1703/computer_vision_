from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='real_gauge_detector',
            executable='real_gauge_detector',
            name='gauge_detector',
            output='screen',
            emulate_tty=True
        ),
    ])
