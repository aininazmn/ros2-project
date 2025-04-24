import threading
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import signal
import random

class Robot(Node):
    def __init__(self):
        super().__init__('robot')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.laser_subscription = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.obstacle_threshold = 1.0
        self.closest_obstacle_distance = float('inf')
        self.obstacle_detected = False
        
        self.exploring = True
        self.approaching_blue = False
        self.blue_detected = False
        
        self.sensitivity = 15
        self.stop_threshold_area = 400000  # adjust roughly withi a radius of 1 meter from the box
        self.last_seen_blue_time = None
        self.last_seen_blue_direction = 0.0

    def image_callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([120 - self.sensitivity, 100, 100])
        upper_blue = np.array([120 + self.sensitivity, 255, 255])
        blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

        contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                frame_center = image.shape[1] // 2
                offset = cx - frame_center

                self.last_seen_blue_time = time.time()
                self.last_seen_blue_direction = -0.003 * offset  # scale factor for turning
                
                self.get_logger().info(f"Blue detected. Area: {area}, Offset: {offset}")

                if area > 500:  # visible enough
                    self.blue_detected = True
                    self.exploring = False
                    self.approaching_blue = True
                    self.move_towards_blue(area)
        else:
            # If not detected but recently seen, continue moving based on last direction
            pass

        cv2.imshow("Camera View", image)
        cv2.waitKey(3)

    def move_towards_blue(self, area):
        cmd = Twist()
        cmd.angular.z = self.last_seen_blue_direction * 1.5  # faster turn toward blue
        if area < self.stop_threshold_area:
            cmd.linear.x = 0.4  # faster approach
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.publisher_.publish(cmd)
            self.get_logger().info("Reached blue box. Stopping.")
            self.approaching_blue = False
            self.blue_detected = False
            self.stop()
            rclpy.shutdown()
            return
        self.publisher_.publish(cmd)

    def continue_toward_last_seen_blue(self):
        # If recently seen (within 2 seconds), continue to move forward & turn
        if self.last_seen_blue_time and (time.time() - self.last_seen_blue_time < 2):
            cmd = Twist()
            cmd.angular.z = self.last_seen_blue_direction
            cmd.linear.x = 0.1
            self.publisher_.publish(cmd)
        else:
            # Otherwise stop
            self.approaching_blue = False
            self.blue_detected = False

    def laser_callback(self, data):
        # Get the minimum distance in front of the robot
        front_ranges = [r for r in data.ranges[0:10] + data.ranges[-10:] if r > 0]
        if front_ranges:
            closest_distance = min(front_ranges)
        else:
            closest_distance = float('inf')

        self.closest_obstacle_distance = closest_distance
        self.obstacle_detected = closest_distance < self.obstacle_threshold


    def avoid_obstacle(self):
        # Step 1: Quick reverse
        reverse_cmd = Twist()
        reverse_cmd.linear.x = -0.3  # faster backup
        reverse_cmd.angular.z = 0.0
        self.publisher_.publish(reverse_cmd)
        time.sleep(0.7)  # shorter reverse time

        # Step 2: Quick left turn
        turn_cmd = Twist()
        turn_cmd.linear.x = 0.0
        turn_cmd.angular.z = 0.7  # faster turn
        self.publisher_.publish(turn_cmd)
        time.sleep(0.9)  # quick turn

        # Step 3: Move forward out of the obstacle zone
        forward_cmd = Twist()
        forward_cmd.linear.x = 0.5  # faster forward recovery
        forward_cmd.angular.z = 0.0
        self.publisher_.publish(forward_cmd)
        time.sleep(1.5)  # move forward for a bit less time


    def explore(self):
        cmd = Twist()
        # If obstacle detected, avoid
        if self.obstacle_detected:
            self.avoid_obstacle()
        else:
            # Slow down as you get closer to obstacles
            if self.closest_obstacle_distance < 1.5:
                cmd.linear.x = 0.2  # slow down
            else:
                cmd.linear.x = 0.5  # full speed
            cmd.angular.z = 0.0
            self.publisher_.publish(cmd)



    def stop(self):
        cmd = Twist()
        self.publisher_.publish(cmd)

def main():
    def signal_handler(sig, frame):
        robot.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    robot = Robot()
    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            if robot.exploring and not robot.blue_detected:
                robot.explore()
                time.sleep(0.5)
            elif robot.approaching_blue:
                # If lost sight of blue temporarily, continue moving toward last known direction
                robot.continue_toward_last_seen_blue()
                time.sleep(0.1)
            else:
                robot.stop()
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

