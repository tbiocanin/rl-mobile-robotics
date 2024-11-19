FROM dustynv/ros:noetic-pytorch-l4t-r32.7.1
ENV DEBIAN_FRONTEND=noninteractive

RUN git clone https://github.com/tbiocanin/rl-mobile-robotics.git

RUN bash -c "source /opt/ros/noetic/setup.bash && cd rl-mobile-robotics/ && catkin_make"

# setting the repo as the default work dir
WORKDIR /rl-mobile-robotics
