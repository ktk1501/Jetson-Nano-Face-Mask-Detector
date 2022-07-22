# Jetson Nano Face Mask Detection

Face mask detector for COVID-19 monitoring implemented on a Jetson Nano. The entire framework performs real-time at a 7 Hz. Any type of webcams or videostreams are available.

# Dependencies

- tensorflow: 1.15.x
- keras
- imutils
- numpy
- opencv-python

# Framework Hierarchy

This detection framework consists of two components:

1. **MTCNN based on TensorRT**: It detects face in the given image.
2. **Publich CNN based model**: It performs classification on the detected faces based on whether or not a mask is present. The output of the model is probability between 0 to 1 (softmax layer). In this repository, I also provide an instruction to convert this model to TensorRT format. **Note that the TensorRT model must be built on the exact hardware you are going to use!**

# Convert h5 Model to TensorRT

Go to [convert_to_trt](convert_to_trt) and follow the steps below:

## model-> h5

```python
from tensorflow import keras
model = keras.models.load_model("mask_detector.model")
model.save("mask_detector.h5")
```

## h5 -> pb

Run [freeze_h5_to_pb.py](convert_to_trt/freeze_h5_to_pb.py)

```bash
pip3 install docopt
python3 freeze_h5_to_pb.py --model="mask_detector.h5" --output="mask_detector.pb"
```

## pb -> uff (uff is a mediate model format that tensorrt 7 proposed)

Run "convert-to-uff.py" in the directory '/usr/lib/python3.?/dist-packages/uff/bin':

```bash
python3 convert-to-uff.py {$ pb file directory}
```

Then, 'mask_detector.uff' will be created in the current directory.

## Run uff File

At the first time you run this uff file, it will serialize it to the tensorrt format. From the next time, it will automatically deserialize the tensorrt model.

| You should follow the steps above to convert plain CNN model to a TensorRT model. **On the other hand, this repository in current form also supports vanilla inference with the plain model.**

# How to Run

```bash
git clone <link-to-this-repository>
cd jetson_nano_face_mask_detector
python face_mask_detector.py
```

# Performance Comparison

I analyzed the inference performance of the compared methods:

1. Vanilla MTCNN + vanilla classification model.
2. TensorRT MTCNN + vanilla classification model.
3. TensorRT MTCNN + TensorRT classification model.
4. [Cythonizing](https://cython.org/) python scripts with option 3.

<img width="476" alt="스크린샷 2020-12-01 오전 12 27 22" src="https://user-images.githubusercontent.com/40379815/100628826-fcc1ab00-336b-11eb-9b0c-90b103686596.png">

# Results

![gif](https://user-images.githubusercontent.com/40379815/100628499-9dfc3180-336b-11eb-9bbc-f17c74947477.gif)

# Related Works

This repository was inspired by the [project of MTCNN TensorRT](https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT).
