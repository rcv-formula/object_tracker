from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class coordinate_converter:
    def __init__(self, node: Node):
        self.node = node
        self.path_recived = False
        self.global_path = []

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.subscription = self.node.create_subscription(Path, 'global_path', self.path_callback, qos_profile)

        # Non-blocking way to wait for path
        self.path_wait_timer = self.node.create_timer(0.1, self.check_path_received)

    def check_path_received(self):
        if self.path_recived:
            self.node.get_logger().info(f"Path received: {len(self.global_path)} points")
            self.path_wait_timer.cancel()  # Stop the timer

    def get_path_length(self, start_idx=None):
        if len(self.global_path) == 0:
            self.node.get_logger().error("Cannot calculate path length: global_path is empty!")
            return 0  # Instead of returning None, return 0

        if start_idx is not None:
            return self._calc_path_distance(start_idx, len(self.global_path) - 1)
        
        return self._calc_path_distance(0, len(self.global_path) - 1)

    def get_global_path(self):
        return self.global_path

    def global_to_frenet_point(self, x, y):
        if len(self.global_path) == 0:
            self.node.get_logger().warn("global_to_frenet_point: No global path available.")
            return [0, 0]  # Return default values if path is missing

        closest_idx = self._get_closest_index(x, y)

        if closest_idx >= len(self.global_path) - 1:
            self.node.get_logger().warn("Closest index is out of bounds!")
            return [0, 0]

        out1 = self._calc_proj(closest_idx, closest_idx + 1, x, y)
        out2 = self._calc_proj(closest_idx - 1, closest_idx, x, y)

        s, d = 0, 0
        self.node.get_logger().info(f"index: {closest_idx}, s: {out1[0]}, d: {out1[1]}")

        if abs(out1[1]) > abs(out2[1]):
            s = out2[0] + self._calc_path_distance(0, closest_idx - 1)
            d = out2[1]
        else:
            s = out1[0] + self._calc_path_distance(0, closest_idx)
            d = out1[1]

        return [s, d]

    def global_to_frenet(self, path_list):
        output = []
        for p in path_list:
            s_d = self.global_to_frenet_point(p[0], p[1])
            v = p[2]
            output.append([s_d[0], s_d[1], v])
        return output

    def frenet_to_global_point(self, s, d):
        if s < 0:
            s += self.get_path_length()
        s = s % self.get_path_length()

        start_idx = self._get_start_path_from_frenet(s)
        next_idx = (start_idx + 1) % len(self.global_path)
        s -= self.get_path_length(start_idx=start_idx)

        start_point = np.array(self.global_path[start_idx][:2])
        next_point = np.array(self.global_path[next_idx][:2])

        path_u_vector = next_point - start_point
        proj_point = start_point + (path_u_vector * s)
        normal_vector = self.rotate_right_90(path_u_vector)
        global_point = proj_point + (normal_vector * d)
        return list(global_point)

    def frenet_to_global(self, path_list: list):
        output = []
        for p in path_list:
            global_point = self.frenet_to_global_point(p[0], p[1])
            output.append(global_point + [p[2]])
        return output

    @staticmethod
    def rotate_right_90(v):
        return np.array([v[1], -v[0]])

    def _get_start_path_from_frenet(self, s):
        idx = 0
        while self._calc_path_distance(0, idx + 1) <= s:
            idx += 1
        return idx

    def _get_closest_index(self, x, y):
        """가장 가까운 글로벌 경로 인덱스를 찾는 함수"""
        if len(self.global_path) == 0:
            self.node.get_logger().warn("Cannot find closest index: Global path is empty!")
            return 0

        idx = 0
        closest_dist = self._calc_distance([x, y], self.global_path[0][:2])
        for i in range(1, len(self.global_path)):
            dist = self._calc_distance([x, y], self.global_path[i][:2])
            if dist < closest_dist:
                idx = i
                closest_dist = dist
        return idx

    def _calc_proj(self, idx, next_idx, x, y):
        path_size = len(self.global_path)
        idx = (idx % path_size + path_size) % path_size
        next_idx = (next_idx % path_size + path_size) % path_size

        pointA = np.array(self.global_path[idx][:2])
        pointB = np.array(self.global_path[next_idx][:2])
        pointC = np.array([x, y])

        vectorA = pointB - pointA
        vectorB = pointC - pointA

        proj_t = np.dot(vectorB, vectorA) / np.dot(vectorA, vectorA)
        proj_point = pointA + (proj_t * vectorA)

        d = np.linalg.norm(proj_point - pointC)
        if np.cross(vectorA, vectorB) > 0:
            d = -d
        s = proj_t * np.dot(vectorA, vectorA)
        return [s, d]

    def _calc_path_distance(self, start_idx, end_idx):
        distance_counter = 0
        if end_idx < start_idx:
            end_idx += len(self.global_path)
        for i in range(start_idx, end_idx):
            distance_counter += self._calc_path_to_path_distance(i)
        return distance_counter

    def _calc_path_to_path_distance(self, idx):
        if idx < 0:
            idx += len(self.global_path)
        idx = idx % len(self.global_path)
        next_idx = (idx + 1) % len(self.global_path)

        cur = self.global_path[idx][:2]
        next = self.global_path[next_idx][:2]
        return self._calc_distance(cur, next)

    def _calc_distance(self, A, B):
        return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    def path_callback(self, msg: Path):
        if not msg.poses:
            self.node.get_logger().warn("Received empty global path!")
            return

        self.global_path = [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in msg.poses]
        self.path_recived = True
        self.node.get_logger().info(f"Received global path with {len(self.global_path)} points")

def main():
    rclpy.init()
    node = Node("coordinate_converter_test_node")
    converter = coordinate_converter(node)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()