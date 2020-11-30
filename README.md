# jetson_nano_face_mask_detector
Face mask detector for COVID-19 on Jetson Nano SBC, fast and accurate. It performs real-time at 7 FPS. You can use any webcam or videostream. 

# Dependencies
- tensorflow: 1.15.x
- keras
- imutils
- numpy
- opencv-python

# How to Run
```bash
git clone <link-to-this-repository>
cd jetson_nano_face_mask_detector
python face_mask_detector.py
```

# Acceleration
It is accelerated through procedures below:
1. Using light face detector MTCNN.
2. Quantizing the model using TensorRT.
3. Cythonizing python run files.
<img width="476" alt="스크린샷 2020-12-01 오전 12 27 22" src="https://user-images.githubusercontent.com/40379815/100628826-fcc1ab00-336b-11eb-9b0c-90b103686596.png">

# Results
![gif](https://user-images.githubusercontent.com/40379815/100628499-9dfc3180-336b-11eb-9bbc-f17c74947477.gif)
