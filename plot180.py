import pandas as pd
import numpy as np
import streamlit as st
import time
from plotly import graph_objects as go
import os
import inspect
from google.protobuf.json_format import MessageToJson
import argparse
from gym_robotable.envs import logging
import plotly.express as px

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer

#assume radians
def normalize_0_180(img):
    normalized_0_180 = np.array(img)*57.2958 + 90
    return normalized_0_180


if __name__ == "__main__":

    st.title('Analyticz')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_file', help='path to protobuf file', default='/media/chrx/0FEC49A4317DA4DA/walkinglogs/robotable_log_2021-01-17-231240')
    args = parser.parse_args()
    logging = logging.RobotableLogging()
    episode_proto = logging.restore_episode(args.log_file)


    times = []
    angles = [[] for i in range(4)]

    for step in range(len(episode_proto.state_action)):

       step_log = episode_proto.state_action[step]
       times.append(str(step_log.time.seconds) + '.' + str(step_log.time.nanos))
       for i in range(4):
           print (step)
           print (step_log.motor_states[i].angle)
           angles[i].append(step_log.motor_states[i].angle)

    #truncate because a bunch of trailing zeros
    angles[0] = angles[0][0:3000]
    angles[1] = angles[1][0:3000]
    angles[2] = angles[2][0:3000]
    angles[3] = angles[3][0:3000]



    avg_0 = normalize_0_180(angles[0])
    avg_1 = normalize_0_180(angles[1])
    avg_2 = normalize_0_180(angles[2])
    avg_3 = normalize_0_180(angles[3])

    avg_0 = anchor(avg_0, 0.8)
    avg_1 = anchor(avg_1, 0.8)
    avg_2 = anchor(avg_2, 0.8)
    avg_3 = anchor(avg_3, 0.8)


    avg_0 = anchor(avg_0, 0.8)
    avg_1 = anchor(avg_1, 0.8)
    avg_2 = anchor(avg_2, 0.8)
    avg_3 = anchor(avg_3, 0.8)


    avg_0 = anchor(avg_0, 0.8)
    avg_1 = anchor(avg_1, 0.8)
    avg_2 = anchor(avg_2, 0.8)
    avg_3 = anchor(avg_3, 0.8)



    np.save('angle_front_right_180', avg_0)
    np.save('angle_front_left_180', avg_1)
    np.save('angle_back_right_180', avg_2)
    np.save('angle_back_left_180', avg_3)



    # Create traces
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=times, y=angles[0],
	            mode='lines',
	            name='Angles 0'))
    fig0.add_trace(go.Scatter(x=times, y=avg_0,
                 mode='lines',
                 name='Norm Moving Average 0'))

    st.plotly_chart(fig0)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=times, y=angles[1],
	            mode='lines',
	            name='Angles 1'))
    fig1.add_trace(go.Scatter(x=times, y=avg_1,
                 mode='lines',
                 name='Norm Moving Average 1'))
    st.plotly_chart(fig1)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=angles[2],
	            mode='lines',
	            name='Angles 2'))
    fig2.add_trace(go.Scatter(x=times, y=avg_2,
                 mode='lines',
                 name='Norm Moving Average 2'))
    st.plotly_chart(fig2)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=angles[3],
	            mode='lines',
	            name='Angles 3'))
    fig3.add_trace(go.Scatter(x=times, y=avg_3,
                 mode='lines',
                 name='Norm Moving Average 3'))
    st.plotly_chart(fig3)

