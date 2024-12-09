#!/usr/bin/env python3

"""
Node for handling the image processing using depthNet inference.
This node gatheres images from the camera, does the process through the depthNet
and publishes the output on the depth_image topic.
"""

import sys
import torch as th
import cv2
import numpy as np

# ROS specific imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()
MODEL_PATH = "/rl-mobile-robotics/src/dqn-jetson-inference/src/models/depth_anything_v2_metric_vkitti_vits.pth"


def create_depth_image_publisher():
    """
    Creating the depth image publisher node so that the output can be send
    """
    try:
        rospy.loginfo_once("Creating depth image publisher...")
        depth_image_publisher = rospy.Publisher(name="depth_node", data_class=Image, queue_size=1)
        return depth_image_publisher
    except:
        rospy.logerr("Could not create the depth image publisher! Exiting...")
        sys.exit()

def publish_depth_image(depth_field, publisher):
    """
    Transform and prepare the message to be in a desired format to be send to the topic.
    """
    ros_img = bridge.cv2_to_imgmsg(depth_field, encoding="passthrough")
    publisher.publish(ros_img)

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720,
                       display_width=224, display_height=224, framerate=30, flip_method=0):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    )

def main():
    """
    The main sequence of this node.
    """
    
    model = th.load(MODEL_PATH, map_location="cuda")
    model.eval()

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Unable to load the camera")
        sys.exit()

    try:
        while True:
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from camera")
                break

            # Preprocess the frame
            resized_frame = cv2.resize(frame, (224, 224))  # Resize to match model input
            input_tensor = th.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize
            input_tensor = input_tensor.to("cuda")  # Send to GPU if available

            # Run the model
            with th.no_grad():
                output = model(input_tensor)

            # Example: Display the model's output
            print(f"Model output: {output.cpu().numpy()}")

            # Show the live camera feed
            cv2.imshow("Camera Feed", frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
    # rospy.spin()
