#!/usr/bin/env python3

import rospy
from mavros_msgs.msg import WaypointList, WaypointReached

class MissionMonitor:
    def __init__(self):
        self.waypoints = []

        rospy.loginfo("Mission monitor iniciado")

        rospy.Subscriber("/rover_argo_N1/Instance1/mavros/mission/waypoints",
                         WaypointList,
                         self.waypoints_cb, queue_size=1)

        rospy.Subscriber("/rover_argo_N1/Instance1/mavros/mission/reached",
                         WaypointReached,
                         self.reached_cb)

    def waypoints_cb(self, data: WaypointList):
        self.waypoints = data.waypoints
        rospy.loginfo(f"Missão recebida com {len(self.waypoints)} waypoints")

    def reached_cb(self, msg):
        wp_seq = msg.wp_seq

        if wp_seq >= len(self.waypoints):
            return

        cmd = self.waypoints[wp_seq].command

        if cmd == 205:
            rospy.loginfo("📸 TAKE PHOTO detectado!")
            # aqui você coloca o que quiser:
            # - salvar timestamp
            # - publicar MQTT
            # - sincronizar sensor
        elif cmd == 16:
            rospy.loginfo(f"Waypoint normal atingido (seq={wp_seq})")

if __name__ == "__main__":
    rospy.init_node("mission_monitor")
    MissionMonitor()
    rospy.spin()
