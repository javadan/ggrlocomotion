"""A proto buffer based logging system for robotable experiments.

The logging system records the time since reset, base position, orientation,
angular velocity and motor information (joint angle, speed, and torque) into a
proto buffer. See logging.proto for more details. The episode_proto is
updated per time step by the environment and saved onto disk for each episode.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import sys
sys.path.append('/opt/gym-robotable/gym_robotable/envs')
import datetime
import os
import time

from gym_robotable.envs import logging_pb2

NUM_MOTORS = 4


def _update_base_state(base_state, values):
  base_state.x = values[0]
  base_state.y = values[1]
  base_state.z = values[2]


def preallocate_episode_proto(episode_proto, max_num_steps):
  """Preallocate the memory for proto buffer.

  Dynamically allocating memory as the protobuf expands causes unexpected delay
  that is not tolerable with locomotion control.

  Args:
    episode_proto: The proto that holds the state/action data for the current
      episode.
    max_num_steps: The max number of steps that will be recorded in the proto.
      The state/data over max_num_steps will not be stored in the proto.
  """
  for _ in range(max_num_steps):
    step_log = episode_proto.state_action.add()
    step_log.info_valid = False
    step_log.time.seconds = 0
    step_log.time.nanos = 0
    for _ in range(NUM_MOTORS):
      motor_state = step_log.motor_states.add()
      motor_state.angle = 0
      motor_state.velocity = 0
      motor_state.torque = 0
      motor_state.action = 0
    _update_base_state(step_log.base_position, [0, 0, 0])
    _update_base_state(step_log.base_orientation, [0, 0, 0])
    _update_base_state(step_log.base_angular_vel, [0, 0, 0])


def update_episode_proto(episode_proto, robotable, action, step):
  """Update the episode proto by appending the states/action of the robotable.

  Note that the state/data over max_num_steps preallocated
  (len(episode_proto.state_action)) will not be stored in the proto.
  Args:
    episode_proto: The proto that holds the state/action data for the current
      episode.
    robotable: The robotable instance. See envs.robotable for details.
    action: The action applied at this time step. The action is an 8-element
      numpy floating-point array.
    step: The current step index.
  """
  max_num_steps = len(episode_proto.state_action)
  if step >= max_num_steps:
    print ("warning about steps and max_num_steps")
    return
  step_log = episode_proto.state_action[step]
  step_log.info_valid = robotable.IsObservationValid()
  time_in_seconds = robotable.GetTimeSinceReset()
  step_log.time.seconds = int(time_in_seconds)
  step_log.time.nanos = int((time_in_seconds - int(time_in_seconds)) * 1e9)

  motor_angles = robotable.GetMotorAngles()
  motor_velocities = robotable.GetMotorVelocities()
  motor_torques = robotable.GetMotorTorques()
  for i in range(robotable.num_motors):
    step_log.motor_states[i].angle = motor_angles[i]
    step_log.motor_states[i].velocity = motor_velocities[i]
    step_log.motor_states[i].torque = motor_torques[i]
    step_log.motor_states[i].action = action[i]

  _update_base_state(step_log.base_position, robotable.GetBasePosition())
  _update_base_state(step_log.base_orientation, robotable.GetBaseRollPitchYaw())
  _update_base_state(step_log.base_angular_vel, robotable.GetBaseRollPitchYawRate())


class RobotableLogging(object):
  """A logging system that records the states/action of the robotable."""

  def __init__(self, log_path=None):
    self._log_path = log_path

  # TODO(jietan): Consider using recordio to write the logs.
  def save_episode(self, episode_proto):
    """Save episode_proto to self._log_path.

    self._log_path is the directory name. A time stamp is the file name of the
    log file. For example, when self._log_path is "/tmp/logs/", the actual
    log file would be "/tmp/logs/yyyy-mm-dd-hh:mm:ss".

    Args:
      episode_proto: The proto that holds the states/action for the current
        episode that needs to be save to disk.
    Returns:
      The full log path, including the directory name and the file name.
    """
    if not self._log_path or not episode_proto.state_action:
      return self._log_path
    if not os.path.exists(self._log_path):
      os.makedirs(self._log_path)
    ts = time.time()
    time_stamp = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d-%H%M%S")
    log_path = os.path.join(self._log_path, "robotable_log_{}".format(time_stamp))
    with open(log_path, "wb") as f:
      f.write(episode_proto.SerializeToString())
      f.close()
    return log_path

  def restore_episode(self, log_path):
    """Restore the episodic proto from the log path.

    Args:
      log_path: The full path of the log file.
    Returns:
      The robotable episode proto.
    """
    with open(log_path, 'rb') as f:
      content = f.read()
      episode_proto = logging_pb2.RobotableEpisode()
      episode_proto.ParseFromString(content)
      return episode_proto
