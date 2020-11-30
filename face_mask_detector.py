"""trt_mtcnn.py

This script demonstrates how to do real-time face detection with
Cython wrapped TensorRT optimized MTCNN engine.
"""

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

import numpy as np
import imutils
import time
import os
import time
import argparse

import cv2
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.mtcnn import TrtMtcnn


WINDOW_NAME = 'TrtMtcnnDemo'
BBOX_COLOR = (0, 255, 0)  # green


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    # parser.add_argument('--usb', type=int, default=0,
    #                    help='USB webcam device id (/dev/video?) [None]')
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    parser.add_argument("-m", "--model", type=str,
                        default="mask_detector.model",
                        help="path to trained face mask detector model")
    args = parser.parse_args()
    return args


maskCnt = 0
maskThrs = 5


def show_faces_and_predict_mask(img, boxes, landmarks, maskNet):
    faces = []
    locs = []
    preds = []
    global maskCnt
    global maskThrs
    # Draw bounding boxes and face landmarks on image.
    for bb, ll in zip(boxes, landmarks):
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        # for j in range(5):
        # cv2.circle(img, (int(ll[j]), int(ll[j+5])), 2, BBOX_COLOR, 2)
        face = img[y1:y2, x1:x2]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        faces.append(face)
        locs.append((x1, y1, x2, y2))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=1)

    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask and mask > 0.95 else "No Mask"
        maskCnt = maskCnt+1 if label == "Mask" else 0

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame

        if maskCnt > maskThrs:
            color = (0, 255, 0)
            cv2.putText(img, "Access Approved", ((int)(200), (int)(453)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)
        else:
            color = (0, 0, 255)
            cv2.putText(img, "Access Denied", ((int)(200), (int)(453)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)
        cv2.putText(img, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)

    return img


def loop_and_detect(cam, mtcnn, minsize, maskNet):
    """Continuously capture images from camera and do face detection."""
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is not None:
            dets, landmarks = mtcnn.detect(img, minsize=minsize)
            print('{} face(s) found'.format(len(dets)))
            img = show_faces_and_predict_mask(img, dets, landmarks, maskNet)
            img = show_fps(img, fps)
            cv2.imshow(WINDOW_NAME, img)
            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)


def main():
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    mtcnn = TrtMtcnn()

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args.model)  # keras model

    open_window(
        WINDOW_NAME, 'Camera TensorRT MTCNN Demo for Jetson Nano',
        cam.img_width, cam.img_height)
    loop_and_detect(cam, mtcnn, args.minsize, maskNet)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
