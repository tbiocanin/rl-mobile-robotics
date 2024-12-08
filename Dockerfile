FROM dustynv/ros:noetic-pytorch-l4t-r32.7.1
ENV DEBIAN_FRONTEND=noninteractive

# downloading the networks needed
RUN bash -c "cd jetson-inference && mkdir data && cd data && mkdir networks && cd networks && cd ../.. && git clone https://github.com/dusty-nv/jetson-utils"
RUN wget https://nvidia.box.com/shared/static/frgbiqeieaja0o8b0eyb87fjbsqd4zup.gz && tar -xzvf frgbiqeieaja0o8b0eyb87fjbsqd4zup.gz -C /jetson-inference/data/networks/

# project repo setup
RUN git clone https://github.com/tbiocanin/rl-mobile-robotics.git
RUN bash -c "source /opt/ros/noetic/setup.bash && cd rl-mobile-robotics/ && catkin_make && git fetch && git pull"

# setting the repo as the default work dir
WORKDIR /rl-mobile-robotics

RUN git fetch origin && git pull

# installing the necessary python3 libs
RUN bash -c "pip3 install cv_bridge"
RUN bash -c "pip3 install Jetson.GPIO"

# fix regarding the missing folders in the jetson-inference repo within the base image
COPY src/dqn-jetson-inference/src/models.json /jetson-inference/data/networks/