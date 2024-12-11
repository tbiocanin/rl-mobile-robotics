#!/usr/bin/env python3

"""
Interface for working with the DQN model optimized with TensorRT
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
ENGINE_PATH = "/rl-mobile-robotics/src/dqn-jetson-inference/src/models/model1.trt"

class DQN_Inference():
    """
    Helper class for working with the DQN model.
    """
    def __init__(self):
        self.engine = self._open_engine()
        self.context = self.engine.create_execution_context()

        self.input_shape = (1, 1, 256, 256)
        self.output_shape = 3
        self.input_data = np.zeros(self.input_shape).astype(np.float32)
        self.output_data = np.empty(self.output_shape, dtype=np.int8)

        self.d_input = cuda.mem_alloc(self.input_data.nbytes)
        self.d_output = cuda.mem_alloc(self.output_data.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()

    def _open_engine(self):

        with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        return engine

    def run_dqn_inference(self, input_image):
        
        cuda.memcpy_htod(self.d_input, input_image)
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.output_data, self.d_output)

        print(self.output_data)
        
        return self.output_data


if __name__ == "__main__":
    pass