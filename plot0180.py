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

def normalize_0_180(img):
    b = (img - np.min(img))/np.ptp(img)
    #normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    normalized_0_180 = (180*(img - np.min(img))/np.ptp(img)).astype(int)   
    return normalized_0_180

def normalize_negative_one(img):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return 2*normalized_input - 1

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

    #get moving averages
    window_size_0=40
    numbers_series_0 = pd.Series(velocities[0])
    windows_0 = numbers_series_0.rolling(window_size_0)
    moving_averages_0 = windows_0.mean()
    moving_averages_list_0 = moving_averages_0.tolist()
    without_nans_0 = moving_averages_list_0[window_size_0 - 1:]

    window_size_1=40
    numbers_series_1 = pd.Series(velocities[1])
    windows_1 = numbers_series_1.rolling(window_size_1)
    moving_averages_1 = windows_1.mean()
    moving_averages_list_1 = moving_averages_1.tolist()
    without_nans_1 = moving_averages_list_1[window_size_1 - 1:]

    window_size_2=40
    numbers_series_2 = pd.Series(velocities[2])
    windows_2 = numbers_series_2.rolling(window_size_2)
    moving_averages_2 = windows_2.mean()
    moving_averages_list_2 = moving_averages_2.tolist()
    without_nans_2 = moving_averages_list_2[window_size_2 - 1:]

    window_size_3=40
    numbers_series_3 = pd.Series(velocities[3])
    windows_3 = numbers_series_3.rolling(window_size_3)
    moving_averages_3 = windows_3.mean()
    moving_averages_list_3 = moving_averages_3.tolist()
    without_nans_3 = moving_averages_list_3[window_size_3 - 1:]

    #normalize between -1 and 1
    avg_0 = np.asarray(without_nans_0)
    avg_1 = np.asarray(without_nans_1)
    avg_2 = np.asarray(without_nans_2)
    avg_3 = np.asarray(without_nans_3)

    avg_0 = normalize_negative_one(avg_0)
    avg_1 = normalize_negative_one(avg_1)
    avg_2 = normalize_negative_one(avg_2)
    avg_3 = normalize_negative_one(avg_3)

    np.save('velocity_front_right', avg_0)
    np.save('velocity_front_left', avg_1)
    np.save('velocity_back_right', avg_2)
    np.save('velocity_back_left', avg_3)
    np.save('times', times)


    #normalize between 0 and 180
    avg_0 = np.asarray(without_nans_0)
    avg_1 = np.asarray(without_nans_1)
    avg_2 = np.asarray(without_nans_2)
    avg_3 = np.asarray(without_nans_3)

    avg_0 = normalize_0_180(avg_0)
    avg_1 = normalize_0_180(avg_1)
    avg_2 = normalize_0_180(avg_2)
    avg_3 = normalize_0_180(avg_3)

    np.save('velocity_front_right_180', avg_0)
    np.save('velocity_front_left_180', avg_1)
    np.save('velocity_back_right_180', avg_2)
    np.save('velocity_back_left_180', avg_3)



    # Create traces
    fig0 = go.Figure()
    fig0.add_trace(go.Scatter(x=times, y=velocities[0],
	            mode='lines',
	            name='Velocities 0'))

    fig0.add_trace(go.Scatter(x=times, y=avg_0.tolist(),
	            mode='lines',
	            name='Norm Moving Average 0'))

    st.plotly_chart(fig0)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=times, y=velocities[1],
	            mode='lines',
	            name='Velocities 1'))
    fig1.add_trace(go.Scatter(x=times, y=avg_1.tolist(),
	            mode='lines',
	            name='Norm Moving Average 1'))

    st.plotly_chart(fig1)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=velocities[2],
	            mode='lines',
	            name='Velocities 2'))
    fig2.add_trace(go.Scatter(x=times, y=avg_2.tolist(),
	            mode='lines',
	            name='Norm Moving Average 2'))

    st.plotly_chart(fig2)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=times, y=velocities[3],
	            mode='lines',
	            name='Velocities 3'))
    fig3.add_trace(go.Scatter(x=times, y=avg_3.tolist(),
	            mode='lines',
	            name='Norm Moving Average 3'))

    st.plotly_chart(fig3)

