#!/usr/bin/env python3

import math

import rospy

from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32, Float32MultiArray
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker


class CircleLateralController(object):
    """
    Steering-only controller for following a circular path.

    Inputs:
        /vicon/jetracer3
        /sensors_and_input_3

    Outputs:
        /steering_angle_3
        /circle_controller_3/radial_error
        /circle_controller_3/heading_error
        /circle_controller_3/target_x
        /circle_controller_3/target_y
        /circle_controller_3/reference_circle
        /circle_controller_3/lookahead_target

    This node never publishes throttle.
    """

    def __init__(self):
        # ----------------------------------------------------------
        # Vehicle and topic configuration
        # ----------------------------------------------------------
        self.car_number = int(
            rospy.get_param("~car_number", 3)
        )

        self.vicon_topic = rospy.get_param(
            "~vicon_topic",
            "/vicon/jetracer{}".format(self.car_number)
        )

        self.sensor_topic = rospy.get_param(
            "~sensor_topic",
            "/sensors_and_input_{}".format(self.car_number)
        )

        self.steering_topic = rospy.get_param(
            "~steering_topic",
            "/steering_angle_{}".format(self.car_number)
        )

        # ----------------------------------------------------------
        # Desired circle
        # ----------------------------------------------------------
        self.center_x = float(
            rospy.get_param("~center_x", 0.0)
        )

        self.center_y = float(
            rospy.get_param("~center_y", 0.0)
        )

        self.radius = float(
            rospy.get_param("~radius", 5.0)
        )

        # +1 = counterclockwise
        # -1 = clockwise
        direction = int(
            rospy.get_param("~direction", 1)
        )

        self.direction = 1 if direction >= 0 else -1

        # ----------------------------------------------------------
        # Pure-pursuit controller parameters
        # ----------------------------------------------------------
        self.wheelbase = float(
            rospy.get_param("~wheelbase", 0.175)
        )

        self.lookahead = float(
            rospy.get_param("~lookahead", 0.6)
        )

        self.max_steering_angle = float(
            rospy.get_param("~max_steering_angle", 0.45)
        )

        self.steering_sign = float(
            rospy.get_param("~steering_sign", 1.0)
        )

        self.yaw_offset = float(
            rospy.get_param("~yaw_offset", 0.0)
        )

        # Delay compensation
        self.delay = float(
            rospy.get_param("~delay", 0.165)
        )

        self.control_rate = float(
            rospy.get_param("~control_rate", 10.0)
        )

        self.vicon_timeout = float(
            rospy.get_param("~vicon_timeout", 0.25)
        )

        # ----------------------------------------------------------
        # State
        # ----------------------------------------------------------
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.velocity = 0.0
        self.yaw_rate = 0.0

        self.have_vicon_pose = False
        self.have_sensor_data = False

        self.last_vicon_time = rospy.Time(0)

        # ----------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------
        self.steering_publisher = rospy.Publisher(
            self.steering_topic,
            Float32,
            queue_size=1
        )

        self.radial_error_publisher = rospy.Publisher(
            "/circle_controller_{}/radial_error".format(
                self.car_number
            ),
            Float32,
            queue_size=1
        )

        self.heading_error_publisher = rospy.Publisher(
            "/circle_controller_{}/heading_error".format(
                self.car_number
            ),
            Float32,
            queue_size=1
        )

        self.target_x_publisher = rospy.Publisher(
            "/circle_controller_{}/target_x".format(
                self.car_number
            ),
            Float32,
            queue_size=1
        )

        self.target_y_publisher = rospy.Publisher(
            "/circle_controller_{}/target_y".format(
                self.car_number
            ),
            Float32,
            queue_size=1
        )

        self.circle_marker_publisher = rospy.Publisher(
            "/circle_controller_{}/reference_circle".format(
                self.car_number
            ),
            Marker,
            queue_size=1,
            latch=True
        )

        self.target_marker_publisher = rospy.Publisher(
            "/circle_controller_{}/lookahead_target".format(
                self.car_number
            ),
            Marker,
            queue_size=1
        )

        # ----------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------
        rospy.Subscriber(
            self.vicon_topic,
            PoseWithCovarianceStamped,
            self.vicon_callback,
            queue_size=1
        )

        rospy.Subscriber(
            self.sensor_topic,
            Float32MultiArray,
            self.sensor_callback,
            queue_size=1
        )

        rospy.sleep(0.5)
        self.publish_circle_marker()

        rospy.loginfo(
            "Circle lateral controller started for car %d",
            self.car_number
        )

        rospy.loginfo(
            "Vicon topic: %s",
            self.vicon_topic
        )

        rospy.loginfo(
            "Steering topic: %s",
            self.steering_topic
        )

        rospy.loginfo(
            "Circle center=(%.2f, %.2f), radius=%.2f m",
            self.center_x,
            self.center_y,
            self.radius
        )

    @staticmethod
    def wrap_angle(angle):
        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    def vicon_callback(self, msg):
        self.robot_x = float(
            msg.pose.pose.position.x
        )

        self.robot_y = float(
            msg.pose.pose.position.y
        )

        q = msg.pose.pose.orientation

        quaternion = (
            q.x,
            q.y,
            q.z,
            q.w
        )

        _, _, measured_yaw = euler_from_quaternion(
            quaternion
        )

        self.robot_yaw = self.wrap_angle(
            measured_yaw + self.yaw_offset
        )

        self.have_vicon_pose = True
        self.last_vicon_time = rospy.Time.now()

    def sensor_callback(self, msg):
        """
        Expected combined sensor layout:

            data[5] = yaw rate
            data[6] = encoder velocity
        """

        if len(msg.data) <= 6:
            rospy.logwarn_throttle(
                2.0,
                "Sensor message contains fewer than seven values."
            )
            return

        self.yaw_rate = float(msg.data[5])
        self.velocity = float(msg.data[6])

        self.have_sensor_data = True

    def vicon_is_valid(self):
        if not self.have_vicon_pose:
            rospy.logwarn_throttle(
                1.0,
                "Waiting for Vicon pose on %s",
                self.vicon_topic
            )
            return False

        age = (
            rospy.Time.now() - self.last_vicon_time
        ).to_sec()

        if age > self.vicon_timeout:
            rospy.logwarn_throttle(
                1.0,
                "Vicon pose on %s is stale: %.3f seconds old",
                self.vicon_topic,
                age
            )
            return False

        return True

    def publish_neutral_steering(self):
        self.steering_publisher.publish(
            Float32(data=0.0)
        )

    def compute_control(self):
        if not self.vicon_is_valid():
            self.publish_neutral_steering()
            return

        robot_x = self.robot_x
        robot_y = self.robot_y
        robot_yaw = self.robot_yaw

        # ----------------------------------------------------------
        # Delay compensation using encoder velocity and IMU yaw rate
        # ----------------------------------------------------------
        if self.have_sensor_data:
            robot_x += (
                math.cos(robot_yaw)
                * self.velocity
                * self.delay
            )

            robot_y += (
                math.sin(robot_yaw)
                * self.velocity
                * self.delay
            )

            robot_yaw = self.wrap_angle(
                robot_yaw
                + self.yaw_rate
                * self.delay
            )

        dx_center = robot_x - self.center_x
        dy_center = robot_y - self.center_y

        measured_radius = math.hypot(
            dx_center,
            dy_center
        )

        if measured_radius < 0.05:
            rospy.logwarn_throttle(
                1.0,
                "Robot is too close to the circle center."
            )

            self.publish_neutral_steering()
            return

        # Current angular position around the circle
        current_theta = math.atan2(
            dy_center,
            dx_center
        )

        # Convert lookahead distance to an angular displacement
        lookahead_angle = (
            self.lookahead / self.radius
        )

        target_theta = (
            current_theta
            + self.direction * lookahead_angle
        )

        target_x = (
            self.center_x
            + self.radius * math.cos(target_theta)
        )

        target_y = (
            self.center_y
            + self.radius * math.sin(target_theta)
        )

        dx_target = target_x - robot_x
        dy_target = target_y - robot_y

        target_distance = math.hypot(
            dx_target,
            dy_target
        )

        target_heading = math.atan2(
            dy_target,
            dx_target
        )

        heading_error = self.wrap_angle(
            target_heading - robot_yaw
        )

        # Pure-pursuit steering law
        steering_angle = math.atan2(
            2.0
            * self.wheelbase
            * math.sin(heading_error),
            max(target_distance, 0.001)
        )

        steering_angle *= self.steering_sign

        steering_angle = max(
            -self.max_steering_angle,
            min(
                self.max_steering_angle,
                steering_angle
            )
        )

        radial_error = (
            measured_radius - self.radius
        )

        self.steering_publisher.publish(
            Float32(data=steering_angle)
        )

        self.radial_error_publisher.publish(
            Float32(data=radial_error)
        )

        self.heading_error_publisher.publish(
            Float32(data=heading_error)
        )

        self.target_x_publisher.publish(
            Float32(data=target_x)
        )

        self.target_y_publisher.publish(
            Float32(data=target_y)
        )

        self.publish_target_marker(
            target_x,
            target_y
        )

    def publish_circle_marker(self):
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "reference_circle"
        marker.id = 0

        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.04

        marker.color.r = 0.8
        marker.color.g = 0.0
        marker.color.b = 0.8
        marker.color.a = 0.9

        number_of_points = 200

        from geometry_msgs.msg import Point

        for index in range(number_of_points + 1):
            angle = (
                2.0
                * math.pi
                * float(index)
                / float(number_of_points)
            )

            point = Point()

            point.x = (
                self.center_x
                + self.radius * math.cos(angle)
            )

            point.y = (
                self.center_y
                + self.radius * math.sin(angle)
            )

            point.z = 0.0

            marker.points.append(point)

        self.circle_marker_publisher.publish(marker)

    def publish_target_marker(self, target_x, target_y):
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()

        marker.ns = "lookahead_target"
        marker.id = 1

        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = target_x
        marker.pose.position.y = target_y
        marker.pose.position.z = 0.0

        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.scale.z = 0.12

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.target_marker_publisher.publish(marker)

    def run(self):
        rate = rospy.Rate(
            self.control_rate
        )

        marker_counter = 0

        while not rospy.is_shutdown():
            self.compute_control()
            marker_counter += 1

            if marker_counter >= 50:
                self.publish_circle_marker()
                marker_counter = 0

            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node(
            "circle_lateral_controller",
            anonymous=False
        )

        controller = CircleLateralController()
        controller.run()

    except rospy.ROSInterruptException:
        pass
