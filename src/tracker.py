#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node

from scipy.linalg import block_diag
from filterpy.kalman import ExtendedKalmanFilter as EKF
from filterpy.common import Q_discrete_white_noise

from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray 

import sys
sys.path.append('/detect_ws/src/coordinate/src')
from coordinate import coordinate_converter

class ObstacleTracker(Node):
    def __init__(self):
        super().__init__('tracker_node')
        self.converter = coordinate_converter(self)
        self.get_logger().info("Obstacle Tracker Initialized")

        # I JUST MADE THIS !!!!!!!!!!!!!!!!!!!
        # 기본값 초기화
        self.global_x = 0.0
        self.global_y = 0.0

        self.globalpath_s = np.array([])
        self.globalpath_d = np.array([])
        self.globalpath_v = np.array([])
        self.track_length = None

        self.dt = 0.1

        self.ekf = EKF(dim_x=3, dim_z=3)
        self.ekf.F = np.array([[1, 0, self.dt],
                               [0, 1, 0],
                               [0, 0, 1]])

        q_s = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.1)
        q_v = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.1)
        q_d = np.array([[0.01]])  # for d

        self.ekf.Q = block_diag(q_s, q_d, q_v)
        self.ekf.H = np.eye(3)
        self.ekf.R = np.diag([0.5, 0.5, 0.5])
        self.ekf.P *= 5

        self.obstacle_detected = False
        self.is_initialized = False

        self.obs_x = self.obs_y = self.obs_vx = self.obs_vy = 0.0

        self.create_subscription(Bool, "/obstacle_detected", self.obstacle_detected_callback, 10)
        self.create_subscription(Odometry, "/opponent_odom", self.obstacle_callback, 10)

        self.predicted_obstacle_pub = self.create_publisher(Odometry, "/predicted_obstacle", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/visualization_marker", 10)

        self.timer = self.create_timer(0.1, self.run)

    def obstacle_detected_callback(self, msg):
        self.obstacle_detected = msg.data

    def obstacle_callback(self, msg):
        if not hasattr(msg.pose, 'pose'):
            return
        self.obs_x = msg.pose.pose.position.x
        self.obs_y = msg.pose.pose.position.y
        self.obs_vx = msg.twist.twist.linear.x
        self.obs_vy = msg.twist.twist.linear.y

    def global_path(self):
        if not self.converter.path_recived or len(self.converter.get_global_path()) == 0:
            self.get_logger().warn("Waiting for global path...")
            return

        self.globalpath = self.converter.get_global_path()
        self.get_logger().info(f"Received global path with {len(self.globalpath)} points")

        output = self.converter.global_to_frenet(self.globalpath)
        if not output or not isinstance(output, list):
            self.get_logger().error("global_to_frenet returned an empty or invalid output")
            return

        try:
            self.globalpath_s = np.array([p[0] for p in output])
            self.globalpath_d = np.array([p[1] for p in output])
            self.globalpath_v = np.array([p[2] for p in output])
        except IndexError as e:
            self.get_logger().error(f"IndexError in processing output: {e}")
            return

        if self.track_length is None:
            self.track_length = self.converter.get_path_length()
            if self.track_length is None:
                self.get_logger().error("Track length is still None!")
            else:
                self.get_logger().info(f"Track length set: {self.track_length}")

    @staticmethod
    def normalize_s(s, track_length):
        if track_length is None or track_length == 0:
            return s
        return s % track_length

    def filter_outlier(self, s_meas, d_meas, threshold_s=1.0, threshold_d=0.5):
        diff_s = abs(self.normalize_s(s_meas - self.ekf.x[0], self.track_length))
        diff_d = abs(d_meas - self.ekf.x[1])
        if diff_s > threshold_s:
            s_meas = self.ekf.x[0]
        if diff_d > threshold_d:
            d_meas = self.ekf.x[1]
        return s_meas, d_meas

    def find_nearest_global_speed(self, s_obs, d_obs):
        if len(self.globalpath_s) == 0 or len(self.globalpath_v) == 0:
            self.get_logger().warn("Global path is not available for speed lookup.")
            return 0.0  # Return 0 velocity if no path is available

        # Compute Euclidean distance to find the nearest point
        distance = np.sqrt((self.globalpath_s - s_obs) ** 2 + (self.globalpath_d - d_obs) ** 2)
        nearest_idx = np.argmin(distance)  # Index of the closest point

        return self.globalpath_v[nearest_idx]

    def publish_predicted_obstacle(self):
        if self.track_length is None:
            self.get_logger().warn("Track length is not set, skipping obstacle prediction publishing.")
            return

        obs = Odometry()
        obs.header.stamp = self.get_clock().now().to_msg()
        obs.header.frame_id = "map"

        # Convert predicted (s, d) to global (x, y)
        self.global_x, self.global_y = self.converter.frenet_to_global_point(self.ekf.x[0], self.ekf.x[1])
        
        obs.pose.pose.position.x = float(self.global_x)
        obs.pose.pose.position.y = float(self.global_y)
        obs.pose.pose.position.z = 0.0  # Assuming ground level
        
        obs.twist.twist.linear.x = float(self.ekf.x[2])  # Predicted velocity
        obs.twist.twist.linear.y = 0.0  # Assuming no lateral velocity
        obs.twist.twist.linear.z = 0.0  # Assuming no vertical velocity

        self.predicted_obstacle_pub.publish(obs)
        self.get_logger().info(f"Published predicted obstacle at ({self.global_x}, {self.global_y}) with velocity {self.ekf.x[2]}")

    def publish_markers(self):
        markers = MarkerArray()
        if self.is_initialized:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.position.x = float(self.global_x)
            marker.pose.position.y = float(self.global_y)
            marker.pose.position.z = 0.0
            markers.markers.append(marker)

        self.marker_pub.publish(markers)

    
    def update(self):
        if self.track_length is None:
            return

        # 측정값 x,y,vx,vy -> s,d,v
        s_meas, d_meas = self.converter.global_to_frenet_point(self.obs_x, self.obs_y)
        s_meas = self.normalize_s(s_meas, self.track_length)
        s_meas, d_meas = self.filter_outlier(s_meas, d_meas, threshold_s=1.0, threshold_d=0.5)

        v_meas = np.hypot(self.obs_vx, self.obs_vy)

        # 측정 벡터 z = [s, d, v]
        z = np.array([s_meas, d_meas, v_meas])
        self.ekf.update(z)
        self.is_initialized = True


    
    def run(self):
        if self.track_length is None:
            self.global_path()

        if not self.obstacle_detected:
            predicted_s = self.ekf.x[0] + self.ekf.x[2] * self.dt
            predicted_d = self.ekf.x[1]
            predicted_v = self.ekf.x[2] + 0.5 * (self.find_nearest_global_speed(predicted_s, predicted_d) - self.ekf.x[2])
            self.ekf.x = np.array([predicted_s, predicted_d, predicted_v])
        else:
            self.update()

        self.ekf.x[0] = self.normalize_s(self.ekf.x[0], self.track_length)
        self.publish_predicted_obstacle()
        self.publish_markers()

    

def main():
    rclpy.init()
    tracker = ObstacleTracker()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
