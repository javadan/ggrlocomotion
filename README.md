#Gallus Gallus Roboticus locomotion training 

Copyleft License

[mirandamoss.com/gallusgallusroboticus](mirandamoss.com/gallusgallusroboticus)

Early implementation of Sim2Real for the quadruped 'Robotable', using pyBullet.  
Was used to evolve robot locomotion in simulation, and extract angles for servos.


Related to [ggrlococloud](https://github.com/javadan/ggrlococloud):  ggrlococloud is the cloud training version.


Superceded by [ggrkmpeditor](https://github.com/javadan/ggrkmpeditor), because when making numerous robots, it turns out it's a lot easier (and faster) to just design some movements visually, and edit and classify them, than it is to evolve neural networks.  

Nice idea for future development of this, would be to implement PGA-MAP-Elites, or make a pygame ui on top of it.  
A ROS2 implementation with gazebo is probably what I'd do, if starting from scratch, in > 2022.  But pyBullet, Gazebo, the now free MuJoCo, all great physics engines.
