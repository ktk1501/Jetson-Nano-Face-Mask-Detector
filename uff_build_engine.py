"""build_engine.py

This script converts a SSD model (pb) to UFF and subsequently builds
the TensorRT engine.

Input : ssd_mobilenet_v[1|2]_[coco|egohands].pb
Output: TRT_ssd_mobilenet_v[1|2]_[coco|egohands].bin
"""


import os
import ctypes
import argparse

import uff
import tensorrt as trt
import graphsurgeon as gs


DIR_NAME = os.path.dirname(__file__)
LIB_FILE = os.path.abspath(os.path.join(DIR_NAME, 'libflattenconcat.so'))
MODEL_SPECS = {
    'mask_detector': {
        'input_pb':   os.path.abspath(os.path.join(
                          DIR_NAME, 'mask_detector.pb')),
        'tmp_uff':    os.path.abspath(os.path.join(
                          DIR_NAME, 'mask_detector.uff')),
        'output_bin': os.path.abspath(os.path.join(
                          DIR_NAME, 'mask_detector.bin'))
    }
}

INPUT_DIMS = (3, 224, 224)
DEBUG_UFF = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, choices=list(MODEL_SPECS.keys()))
    args = parser.parse_args()

    # initialize
    if trt.__version__[0] < '7':
        ctypes.CDLL(LIB_FILE)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # compile the model into TensorRT engine
    model = args.model
    spec = MODEL_SPECS[model]
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 28
        builder.max_batch_size = 1
        builder.fp16_mode = True

        parser.register_input('Input', INPUT_DIMS)
        #parser.register_output('MarkOutput_0')
        parser.parse(spec['tmp_uff'], network)
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(spec['output_bin'], 'wb') as f:
            f.write(buf)


if __name__ == '__main__':
    main()
