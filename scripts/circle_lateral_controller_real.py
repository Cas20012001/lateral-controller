#!/usr/bin/env python3

import math
import time

import rospy

from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker


class CircleLateralControllerReal(object):

    def __init__(self):

        # ============================================================
        # Vehicle
        # ============================================================

        self.car_number = int(
            rospy.get_param("~car_number", 2)
        )

        # ============================================================
        # Topics
        # ============================================================

        self.vicon_topic = rospy.get_param(
            "~vicon_topic",
            "/vicon/jetracer{}".format(self.car_number)
        )

        self.velocity_topic = rospy.get_param(
            "~velocity_topic",
            "/vicon_velocity_{}".format(self.car_number)
        )

        self.steering_topic = rospy.get_param(
            "~steering_topic",
            "/steering_angle_{}".format(self.car_number)
        )

        # ============================================================
        # Desired circle
        # ============================================================

        self.center_x = float(
            rospy.get_param("~center_x", 0.0)
        )

        self.center_y = float(
            rospy.get_param("~center_y", 0.0)
        )

        self.radius = float(
            rospy.get_param("~radius", 2.5)
        )

        # +1 = counterclockwise
        # -1 = clockwise
        direction = int(
            rospy.get_param("~direction", 1)
        )

        self.direction = 1 if direction >= 0 else -1

        # ============================================================
        # Pure-pursuit parameters
        # ============================================================

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

        # Correction between Vicon rigid-body orientation
        # and actual vehicle forward direction.
        self.yaw_offset = float(
            rospy.get_param("~yaw_offset", 0.0)
        )

        # ============================================================
        # Delay compensation
        # ============================================================

        self.delay = float(
            rospy.get_param("~delay", 0.165)
        )

        # ============================================================
        # Execution
        # ============================================================

        self.control_rate = float(
            rospy.get_param("~control_rate", 10.0)
        )

        self.vicon_timeout = float(
            rospy.get_param("~vicon_timeout", 0.25)
        )

        self.velocity_timeout = float(
            rospy.get_param("~velocity_timeout", 0.25)
        )

        # ============================================================
        # State
        # ============================================================

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.velocity = 0.0
        self.yaw_rate = 0.0

        self.have_vicon_pose = False
        self.have_velocity = False

        self.last_vicon_receive_time = None
        self.last_velocity_receive_time = None

        # Used for yaw-rate derivation from Vicon
        self.previous_yaw = None
        self.previous_vicon_stamp = None

        # ============================================================
        # Publishers
        # ============================================================

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

        # ============================================================
        # Subscribers
        # ============================================================

        rospy.Subscriber(
            self.vicon_topic,
            PoseWithCovarianceStamped,
            self.vicon_callback,
            queue_size=1
        )

        rospy.Subscriber(
            self.velocity_topic,
            Float32,
            self.velocity_callback,
            queue_size=1
        )

        rospy.sleep(0.5)

        self.publish_circle_marker()

        # ============================================================
        # Startup information
        # ============================================================

        rospy.loginfo(
            "Real circle lateral controller started for car %d",
            self.car_number
        )

        rospy.loginfo(
            "Vicon pose topic: %s",
            self.vicon_topic
        )

        rospy.loginfo(
            "Vicon velocity topic: %s",
            self.velocity_topic
        )

        rospy.loginfo(
            "Steering output topic: %s",
            self.steering_topic
        )

        rospy.loginfo(
            "Circle center=(%.3f, %.3f), radius=%.3f m",
            self.center_x,
            self.center_y,
            self.radius
        )

    # ================================================================
    # Helpers
    # ================================================================

    @staticmethod
    def wrap_angle(angle):

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    # ================================================================
    # Vicon callback
    # ================================================================

    def vicon_callback(self, msg):

        self.robot_x = float(
            msg.pose.pose.position.x
        )

        self.robot_y = float(
            msg.pose.pose.position.y
        )

        q = msg.pose.pose.orientation

        sin_yaw = 2.0 * (
            q.w * q.z
            + q.x * q.y
        )

        cos_yaw = 1.0 - 2.0 * (
            q.y * q.y
            + q.z * q.z
        )

        measured_yaw = math.atan2(
            sin_yaw,
            cos_yaw
        )

        yaw = self.wrap_angle(
            measured_yaw + self.yaw_offset
        )

        # ------------------------------------------------------------
        # Derive yaw rate directly from Vicon orientation
        # ------------------------------------------------------------

        stamp = msg.header.stamp.to_sec()

        if stamp <= 0.0:
            stamp = rospy.Time.now().to_sec()

        if (
            self.previous_yaw is not None
            and self.previous_vicon_stamp is not None
        ):

            dt = stamp - self.previous_vicon_stamp

            if 0.0 < dt < 0.2:

                yaw_difference = self.wrap_angle(
                    yaw - self.previous_yaw
                )

                self.yaw_rate = (
                    yaw_difference / dt
                )

        self.previous_yaw = yaw
        self.previous_vicon_stamp = stamp

        self.robot_yaw = yaw

        self.have_vicon_pose = True

        # Monotonic wall time for timeout checking
        self.last_vicon_receive_time = time.monotonic()

    # ================================================================
    # Vicon-derived longitudinal velocity
    # ================================================================

    def velocity_callback(self, msg):

        self.velocity = float(msg.data)

        self.have_velocity = True

        self.last_velocity_receive_time = time.monotonic()

    # ================================================================
    # Input validity
    # ================================================================

    def inputs_are_valid(self):

        # ------------------------------------------------------------
        # Raw Vicon pose
        # ------------------------------------------------------------

        if not self.have_vicon_pose:

            rospy.logwarn_throttle(
                1.0,
                "Waiting for Vicon pose on %s",
                self.vicon_topic
            )

            return False

        vicon_age = (
            time.monotonic()
            - self.last_vicon_receive_time
        )

        if vicon_age > self.vicon_timeout:

            rospy.logwarn_throttle(
                1.0,
                "Vicon pose stale: %.3f seconds old",
                vicon_age
            )

            return False

        # ------------------------------------------------------------
        # Vicon-derived velocity
        # ------------------------------------------------------------

        if not self.have_velocity:

            rospy.logwarn_throttle(
                1.0,
                "Waiting for velocity on %s",
                self.velocity_topic
            )

            return False

        velocity_age = (
            time.monotonic()
            - self.last_velocity_receive_time
        )

        if velocity_age > self.velocity_timeout:

            rospy.logwarn_throttle(
                1.0,
                "Vicon velocity stale: %.3f seconds old",
                velocity_age
            )

            return False

        return True

    # ================================================================
    # Safe steering output
    # ================================================================

    def publish_neutral_steering(self):

        self.steering_publisher.publish(
            Float32(data=0.0)
        )

    # ================================================================
    # Pure-pursuit control
    # ================================================================

    def compute_control(self):

        if not self.inputs_are_valid():

            self.publish_neutral_steering()

            return

        robot_x = self.robot_x
        robot_y = self.robot_y
        robot_yaw = self.robot_yaw

        # ============================================================
        # Delay compensation
        #
        # Predict vehicle pose delay seconds into the future.
        #
        # velocity: from Vicon bridge
        # yaw_rate: derived directly from Vicon orientation
        # ============================================================

        if self.delay > 0.0:

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

        # ============================================================
        # Position relative to circle centre
        # ============================================================

        dx_center = (
            robot_x - self.center_x
        )

        dy_center = (
            robot_y - self.center_y
        )

        measured_radius = math.hypot(
            dx_center,
            dy_center
        )

        if measured_radius < 0.05:

            rospy.logwarn_throttle(
                1.0,
                "Robot is too close to circle centre."
            )

            self.publish_neutral_steering()

            return

        # ============================================================
        # Current angular position around circle
        # ============================================================

        current_theta = math.atan2(
            dy_center,
            dx_center
        )

        # ============================================================
        # Lookahead point on desired circle
        # ============================================================

        lookahead_angle = (
            self.lookahead / self.radius
        )

        target_theta = (
            current_theta
            + self.direction * lookahead_angle
        )

        target_x = (
            self.center_x
            + self.radius
            * math.cos(target_theta)
        )

        target_y = (
            self.center_y
            + self.radius
            * math.sin(target_theta)
        )

        # ============================================================
        # Relative target geometry
        # ============================================================

        dx_target = (
            target_x - robot_x
        )

        dy_target = (
            target_y - robot_y
        )

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

        # ============================================================
        # Pure-pursuit steering law
        # ============================================================

        steering_angle = math.atan2(
            2.0
            * self.wheelbase
            * math.sin(heading_error),
            max(target_distance, 0.001)
        )

        steering_angle *= (
            self.steering_sign
        )

        steering_angle = max(
            -self.max_steering_angle,
            min(
                self.max_steering_angle,
                steering_angle
            )
        )

        # ============================================================
        # Diagnostics
        # ============================================================

        radial_error = (
            measured_radius
            - self.radius
        )

        # ============================================================
        # Publish
        # ============================================================

        self.steering_publisher.publish(
            Float32(
                data=float(steering_angle)
            )
        )

        self.radial_error_publisher.publish(
            Float32(
                data=float(radial_error)
            )
        )

        self.heading_error_publisher.publish(
            Float32(
                data=float(heading_error)
            )
        )

        self.target_x_publisher.publish(
            Float32(
                data=float(target_x)
            )
        )

        self.target_y_publisher.publish(
            Float32(
                data=float(target_y)
            )
        )

        self.publish_target_marker(
            target_x,
            target_y
        )

        rospy.loginfo_throttle(
            1.0,
            "x=%.2f y=%.2f v=%.2f "
            "radial_error=%.3f heading_error=%.3f "
            "steering=%.3f",
            self.robot_x,
            self.robot_y,
            self.velocity,
            radial_error,
            heading_error,
            steering_angle
        )

    # ================================================================
    # RViz reference circle
    # ================================================================

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

        for index in range(
            number_of_points + 1
        ):

            angle = (
                2.0
                * math.pi
                * float(index)
                / float(number_of_points)
            )

            point = Point()

            point.x = (
                self.center_x
                + self.radius
                * math.cos(angle)
            )

            point.y = (
                self.center_y
                + self.radius
                * math.sin(angle)
            )

            point.z = 0.0

            marker.points.append(
                point
            )

        self.circle_marker_publisher.publish(
            marker
        )

    # ================================================================
    # RViz target point
    # ================================================================

    def publish_target_marker(
        self,
        target_x,
        target_y
    ):

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

        self.target_marker_publisher.publish(
            marker
        )

    # ================================================================
    # Main loop
    # ================================================================

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
            "circle_lateral_controller_real",
            anonymous=False
        )

        controller = (
            CircleLateralControllerReal()
        )

        controller.run()

    except rospy.ROSInterruptException:

        pass
