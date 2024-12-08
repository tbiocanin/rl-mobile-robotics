#!/usr/bin/env python3

"""
Node for handling the image processing using depthNet inference.
This node gatheres images from the camera, does the process through the depthNet
and publishes the output on the depth_image topic.
"""

import sys
import argparse

# ROS specific imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Jetson specific imports
from jetson_inference import depthNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log, cudaToNumpy
from depthnet_utils import depthBuffers

bridge = CvBridge()

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

    depth_numpy = cudaToNumpy(depth_field)
    ros_img = bridge.cv2_to_imgmsg(depth_numpy, encoding="passthrough")

    publisher.publish(ros_img)


def main():
    """
    The main sequence of this node.
    """

    # cv bridge object needed for image transformations
    MODEL_PATH = "/jetson-inference/data/networks/MonoDepth-FCN-Mobilenet/monodepth_fcn_mobilenet.onnx"
    DEFAULT_INPUT = "csi://0"

    ros_param_network = rospy.get_param("network", MODEL_PATH)
    ros_param_input = rospy.get_param("input", DEFAULT_INPUT)
    # parse the command line
    parser = argparse.ArgumentParser(description="Mono depth estimation on a video/image stream using depthNet DNN.", 
                                    formatter_class=argparse.RawTextHelpFormatter, 
                                    epilog=depthNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default=ros_param_input, nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default=ros_param_network, help="pre-trained model to load, see below for options")
    parser.add_argument("--visualize", type=str, default="input,depth", help="visualization options (can be 'input' 'depth' 'input,depth'")
    parser.add_argument("--depth-size", type=float, default=1.0, help="scales the size of the depth map visualization, as a percentage of the input size (default is 1.0)")
    parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
    parser.add_argument("--colormap", type=str, default="viridis-inverted", help="colormap to use for visualization (default is 'viridis-inverted')",
                                    choices=["inferno", "inferno-inverted", "magma", "magma-inverted", "parula", "parula-inverted", 
                                            "plasma", "plasma-inverted", "turbo", "turbo-inverted", "viridis", "viridis-inverted"])

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    # needs an explicit --network argument passed (the path to the net within the docker image)
    print("NETWORK PARAM: ", args.network)
    net = depthNet(ros_param_network, sys.argv)
    buffers = depthBuffers(args)

    input = videoSource("csi://0", argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)

    # need to do this only once
    depth_field = net.GetDepthField()
    rospy.init_node("DepthNet_publisher")
    depth_image_publisher = create_depth_image_publisher()

    while True:
        img_input = input.Capture()

        if img_input is None:
            continue

        buffers.Alloc(img_input.shape, img_input.format)
        net.Process(img_input, buffers.depth)
        cudaDeviceSynchronize()
        net.PrintProfilerTimes()

        # publish the processed image
        publish_depth_image(depth_field, depth_image_publisher)

        if not input.IsStreaming() or not output.IsStreaming():
            break

if __name__ == "__main__":
    main()
