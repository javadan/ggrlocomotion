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
from numpy import linalg as LA

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

def normalize_0_180(img):
    b = (img - np.min(img))/np.ptp(img)
    normalized_0_180 = (180*(img - np.min(img))/np.ptp(img)).astype(int)   
    return normalized_0_180


def normalize(a):
    return (a - np.min(a))/np.ptp(a)


def normalize_negative_one(img):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return 2*normalize(img) - 1

def remap(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min



def anchor(signal, weight):
    buffer = []
    last = signal[0]
    for i in signal:
        smoothed_val = last * weight + (1 - weight) * i
        buffer.append(smoothed_val)
        last = smoothed_val
    return buffer


if __name__ == "__main__":

    st.title('Analyticz')

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_file', help='path to protobuf file', default='/media/chrx/0FEC49A4317DA4DA/walkinglogs/robotable_log_2021-01-17-231240')
    args = parser.parse_args()
    logging = logging.RobotableLogging()
    episode_proto = logging.restore_episode(args.log_file)


    times = []
    velocities = [[] for i in range(4)]

    for step in range(len(episode_proto.state_action)):

       step_log = episode_proto.state_action[step]
       times.append(str(step_log.time.seconds) + '.' + str(step_log.time.nanos))

       for i in range(4):
           velocities[i].append(step_log.motor_states[i].velocity)

    #truncate because a bunch of trailing zeros
    velocities[0] = velocities[0][0:3000]
    velocities[1] = velocities[1][0:3000]
    velocities[2] = velocities[2][0:3000]
    velocities[3] = velocities[3][0:3000]
    times = times[0:3000]



    #normalize from -1 to 1
    avg_0 = anchor(velocities[0], 0.9)
    avg_1 = anchor(velocities[1], 0.9)
    avg_2 = anchor(velocities[2], 0.9)
    avg_3 = anchor(velocities[3], 0.9)

#    avg_0 = avg_0/LA.norm(avg_0)
#    avg_1 = avg_1/LA.norm(avg_1)
#    avg_2 = avg_2/LA.norm(avg_2)
#    avg_3 = avg_3/LA.norm(avg_3)
#
#    avg_0 = normalize_negative_one(avg_0)
#    avg_1 = normalize_negative_one(avg_1)
#    avg_2 = normalize_negative_one(avg_2)
#    avg_3 = normalize_negative_one(avg_3)

    avg_0 = remap(avg_0, np.min(avg_0), np.max(avg_0), -1, 1)
    avg_1 = remap(avg_1, np.min(avg_1), np.max(avg_1), -1, 1)
    avg_2 = remap(avg_2, np.min(avg_2), np.max(avg_2), -1, 1)
    avg_3 = remap(avg_3, np.min(avg_3), np.max(avg_3), -1, 1)

    np.save('velocity_front_right', avg_0)
    np.save('velocity_front_left', avg_1)
    np.save('velocity_back_right', avg_2)
    np.save('velocity_back_left', avg_3)
    np.save('times', times)



    #normalize between 0 and 180

    avg_00 = anchor(velocities[0], 0.9)
    avg_11 = anchor(velocities[1], 0.9)
    avg_22 = anchor(velocities[2], 0.9)
    avg_33 = anchor(velocities[3], 0.9)

    avg_00 = normalize_0_180(avg_00)
    avg_11 = normalize_0_180(avg_11)
    avg_22 = normalize_0_180(avg_22)
    avg_33 = normalize_0_180(avg_33)

    np.save('velocity_front_right_180', avg_00)
    np.save('velocity_front_left_180', avg_11)
    np.save('velocity_back_right_180', avg_22)
    np.save('velocity_back_left_180', avg_33)






    # Create traces
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=times, y=velocities[0],
	            mode='lines',
	            name='Velocities 0'))

    fig0.add_trace(go.Scatter(x=times, y=avg_0,
	            mode='lines',
	            name='Norm Moving Average 0'))
   
#    fig0.add_trace(go.Scatter(x=times, y=avg_00,
#	            mode='lines',
#	            name='Norm180'))

    st.plotly_chart(fig0)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=times, y=velocities[1],
	            mode='lines',
	            name='Velocities 1'))
    fig1.add_trace(go.Scatter(x=times, y=avg_1,
	            mode='lines',
	            name='Norm Moving Average 1'))
#    fig1.add_trace(go.Scatter(x=times, y=avg_11,
#                    mode='lines',
#                    name='Norm1801'))


    st.plotly_chart(fig1)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=velocities[2],
	            mode='lines',
	            name='Velocities 2'))
    fig2.add_trace(go.Scatter(x=times, y=avg_2,
	            mode='lines',
	            name='Norm Moving Average 2'))

#    fig2.add_trace(go.Scatter(x=times, y=avg_22,
#                    mode='lines',
#                    name='Norm1802'))





    st.plotly_chart(fig2)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=velocities[3],
	            mode='lines',
	            name='Velocities 3'))
    fig3.add_trace(go.Scatter(x=times, y=avg_3,
	            mode='lines',
	            name='Norm Moving Average 3'))
#    fig3.add_trace(go.Scatter(x=times, y=avg_33,
#                    mode='lines',
#                    name='Norm1803'))
#

    st.plotly_chart(fig3)

