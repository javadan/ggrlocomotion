import time
import numpy as np

from board import SCL, SDA
import busio

from adafruit_pca9685 import PCA9685

from adafruit_motor import servo

i2c = busio.I2C(SCL, SDA)

pca = PCA9685(i2c, reference_clock_speed=25630710)
pca.frequency = 50

#servo0 = servo.ContinuousServo(pca.channels[0], min_pulse=685, max_pulse=2280)
#servo1 = servo.ContinuousServo(pca.channels[1], min_pulse=755, max_pulse=2095)
#servo2 = servo.ContinuousServo(pca.channels[2], min_pulse=700, max_pulse=2140)
#servo3 = servo.ContinuousServo(pca.channels[3], min_pulse=705, max_pulse=2105)

velocity_front_right = np.load('velocity_front_right.npy')
velocity_front_left = np.load('velocity_front_left.npy')
velocity_back_right = np.load('velocity_back_right.npy')
velocity_back_left = np.load('velocity_back_left.npy')
#times = np.load('times.npy')

#reverse left motors
velocity_front_left = -velocity_front_left
velocity_back_left = -velocity_back_left


print (velocity_front_right.size)
print (velocity_front_left.size)
print (velocity_back_right.size)
print (velocity_back_left.size)
#print (times.size)

#for time in times:
#    print(time)

for i in range(velocity_front_right.size):
    servo0.throttle(velocity_front_right[i])
    servo1.throttle(velocity_front_left[i])
    servo2.throttle(velocity_back_right[i])
    servo3.throttle(velocity_back_left[i])
    time.sleep(0.01)
    #print (velocity_front_right[i])
    #print (velocity_front_left[i])
    #print (velocity_back_right[i])
    #print (velocity_back_left[i])

