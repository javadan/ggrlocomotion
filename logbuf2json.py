import os
import inspect
from google.protobuf.json_format import MessageToJson
import pandas as pd
from pandas.io.json import json_normalize



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import argparse
from gym_robotable.envs import logging

if __name__ == "__main__":



    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_file', help='path to protobuf file', default='/media/chrx/0FEC49A4317DA4DA1/logs/robotable_log_2020-12-29-191602')
    args = parser.parse_args()
    logging = logging.RobotableLogging()
    episode_proto = logging.restore_episode(args.log_file)

    jsonObj = MessageToJson(episode_proto)

    pd.read_json(jsonObj)
    print(jsonObj)

#    for step in range(len(episode_proto.state_action)):

#       step_log = episode_proto.state_action[step]

#       for i in range(4):
#           print(step_log.motor_states[i].angle)
#           print(step_log.motor_states[i].velocity)
#           print(step_log.motor_states[i].torque)
#           print(step_log.motor_states[i].action)

   # for step in range(max_num_steps):

    #    step_log = episode_proto.state_action[step]

     #   for i in range(4):
      #      print("servo" + str(i+1) + ".throttle = " + str( step_log.motor_states[i].torque / max / 2))
       #     print("time.sleep(0.02)")





