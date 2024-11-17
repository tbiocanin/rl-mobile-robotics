FROM dustynv/ros:noetic-pytorch-l4t-r32.7.1
ENV DEBIAN_FRONTEND=noninteractive

RUN git clone http://github.com/tbiocanin/rl-mobile-robotics

# setting the repo as the default work dir
WORKDIR /rl-mobile-robotics

# build the project
RUN cd rl-mobile-robotics && catkin_make
