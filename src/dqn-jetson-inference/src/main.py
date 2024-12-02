#!/usr/bin/env python3
"""
Main script for running the inference models on jetson nano
"""

import sys
import argparse
import time

from jetson_inference import depthNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log, cudaToNumpy
from depthnet_utils import depthBuffers

# parse the command line
parser = argparse.ArgumentParser(description="Mono depth estimation on a video/image stream using depthNet DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=depthNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-mobilenet", help="pre-trained model to load, see below for options")
parser.add_argument("--visualize", type=str, default="input,depth", help="visualization options (can be 'input' 'depth' 'input,depth'")
parser.add_argument("--depth-size", type=float, default=1.0, help="scales the size of the depth map visualization, as a percentage of the input size (default is 1.0)")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--colormap", type=str, default="viridis-inverted", help="colormap to use for visualization (default is 'viridis-inverted')",
                                  choices=["inferno", "inferno-inverted", "magma", "magma-inverted", "parula", "parula-inverted", 
                                           "plasma", "plasma-inverted", "turbo", "turbo-inverted", "viridis", "viridis-inverted"])

if __name__ == "__main__":

    try:
	    args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    # needs an explicit --network argument passed (the path to the net within the docker image)
    net = depthNet(args.network, sys.argv)
    buffers = depthBuffers(args)

    input = videoSource(args.input, argv = sys.argv)
    output = videoOutput(args.output, argv=sys.argv)

    # need to do this only once
    depth_field = net.GetDepthField()
    depth_numpy = cudaToNumpy(depth_field)

    # 224x224
    print("Depth field resolution is: ", depth_field.width, depth_field.height)
    # this should be transformed into a publisher

    while True:
        img_input = input.Capture()

        if img_input is None:
            continue

        buffers.Alloc(img_input.shape, img_input.format)

        net.Process(img_input, buffers.depth)

        cudaDeviceSynchronize()
        net.PrintProfilerTimes()

        if not input.IsStreaming() or not output.IsStreaming():
            break

