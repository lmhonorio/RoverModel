#!/usr/bin/env python3

import math
from pymavlink import mavutil

# Classe para formatar itens da missão
class Mission_item:
    def __init__(self, i, current, x, y, z):
        self.seq = i
        self.frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
        self.command = mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
        self.current = current
        self.auto = 1
        self.param1 = 0.0
        self.param2 = 2.00
        self.param3 = 2.00
        self.param4 = 0
        self.param5 = int(x)
        self.param6 = int(y)
        self.param7 = int(z)
        self.mission_type = 0

# Armar o Drone
def arm(the_connection):
    print("Arming")
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                         mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    #ack(the_connection, "COMMAND_ACK")

# Enviar a missão para o drone
def upload_mission(the_connection, mission_items):
    n = len(mission_items)
    print("Sending message out")
    
    the_connection.mav.mission_count_send(the_connection.target_system, the_connection.target_component, n, 0)
    
    for i, waypoint in enumerate(mission_items):
        #print(f"Creating waypoint {waypoint.seq}: ({waypoint.param5}, {waypoint.param6}, {waypoint.param7})")
        
        the_connection.mav.mission_item_int_send(the_connection.target_system, 
                                                 the_connection.target_component, 
                                                 waypoint.seq, 
                                                 waypoint.frame, 
                                                 waypoint.command, 
                                                 waypoint.current, 
                                                 waypoint.auto, 
                                                 waypoint.param1, 
                                                 waypoint.param2, 
                                                 waypoint.param3, 
                                                 waypoint.param4, 
                                                 waypoint.param5, 
                                                 waypoint.param6, 
                                                 waypoint.param7, 
                                                 waypoint.mission_type)
        
        #if i < n - 1:
            #ack(the_connection, "MISSION_REQUEST")

def start_mission(the_connection):
    print("Mission Start")
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component, 
                                         mavutil.mavlink.MAV_CMD_MISSION_START, 0, 0, 0, 0, 0, 0, 0, 0)
    #ack(the_connection, "COMMAND_ACK")

""" def ack(the_connection, keyword):
    print("Before sending %s" % keyword)
    # the_connection.mav.ping_send(the_connection.target_system, the_connection.target_component)
    print("After sending %s" % keyword)
    print("Message read " + str(the_connection.recv_match(type=keyword, blocking=True))) """

# Gerar a missão com base em pontos de missão
def generate_mission(mission_points, robo):
    robo = int(robo)-1
    porta_base = 14550
    porta_robo = porta_base + (robo) * 10
    print(porta_robo)

    connection_string = f'udp:127.0.0.1:{porta_robo}'
    the_connection = mavutil.mavlink_connection(connection_string)

    while the_connection.target_system == 0:
        print("Checking heartbeat")
        the_connection.wait_heartbeat()
        print("Heartbeat from system (system %u component %u)" % (the_connection.target_system, the_connection.target_component))
    arm(the_connection)
    mission_waypoints = []
    counter = 0
    #print(mission_points)
    # mission_points = mission_points[:50]
    for point in mission_points:
        mission_waypoints.append(Mission_item(counter, 0, point[0] * 10 ** 7, point[1] * 10 ** 7, 0))
        counter += 1
    print(mission_points)
    
    upload_mission(the_connection, mission_waypoints)
    #ack(the_connection, "MISSION_ACK")
    start_mission(the_connection)

    the_connection.close()
    print("Conexão encerrada após envio da missão.")

