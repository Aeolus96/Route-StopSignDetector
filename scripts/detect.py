#!/usr/bin/env python3
########################################################################
import rospy  # ROS API
from sensor_msgs.msg import Image  # Image message
from std_msgs.msg import UInt8, UInt32  # UInt8(used as bool) and UInt32(used as +ve number) message
from ultralytics import YOLO  # YOLOv8
import numpy as np  # numpy
import cv2  # OpenCV version 2.x (ROS limitation)
from cv_bridge import CvBridge, CvBridgeError  # ROS to/from OpenCV API
from dynamic_reconfigure.server import Server  # ROS Parameter server for debug
from ltu_actor_route_stop_sign_detector.cfg import StopSignDetectConfig  # packageName.cfg

########################################################################
### Global Variables:
global enabled  # On/Off for the entire code. If no subscribers are available, no need to run this node
enabled = False
global config_  # Dynamic reconfiguration holder
global bridge  # ROS-CV bridge
bridge = CvBridge()
global model_path  # Get yolov8 model's path
global model  # YOLOv8 model loaded globally

########################################################################
### Functions:


# Make a local global copy of the parameter configuration
def dyn_rcfg_cb(config, level):
    global config_
    config_ = config
    return config


# Image callback - Converts ROS Image to OpenCV Image and feeds it to the YOLO layer
def get_image(Image):
    if enabled:
        global bridge
        try:  # convert ros_image into an opencv-compatible image
            cv_image = bridge.imgmsg_to_cv2(Image, "rgb8")
        except CvBridgeError as e:
            print(e)
        # Now cv_image is a standard OpenCV matrice
        find_stop_sign(resize_image(cv_image))
    return


# Resize cv_image to desired size with black letterbox
# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def resize_image(img, size=(640, 640)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1
    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)
    dif = h if h > w else w
    interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC
    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2
    if len(img.shape) == 2:  # Grayscale images
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w] = img[:h, :w]
    else:  # 3-channel color images
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos : y_pos + h, x_pos : x_pos + w, :] = img[:h, :w, :]
    return cv2.resize(mask, size, interpolation)


# Use YOLOv8 to predict stop signs
def find_stop_sign(cv_image):
    show_image = config_.debug # Get bool from ROS dynrcfg
    
    results = model.predict( # Run YOLOv8 prediction with
        source=cv_image,  # Image source
        imgsz=640,  # Image size (square)
        conf=0.7,  # Confidence threshold
        iou=0.75,  # Overlap threshold
        show=show_image,  # Show the results <<<<<<<<TEST
        device="0",  # CUDA device
    )

    # Use results to determine class and size
    # results.boxes (Properties):
    # boxes.xyxy  # box with xyxy format, (N, 4)
    # boxes.xywh  # box with xywh format, (N, 4)
    # boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    # boxes.xywhn  # box with xywh format but normalized, (N, 4)
    # boxes.conf  # confidence score, (N, 1)
    # boxes.cls  # cls, (N, 1)
    # boxes.data  # raw bboxes tensor, (N, 6) or boxes.boxes

    # Get the bounding boxes and class labels for all detected objects
    boxes = results.xyxy[0].numpy()  # Data cast to numpy format
    labels = results.names # Direct class labels access

    # Iterate over all detected objects
    for box in boxes:
        print(box)
        # Get the class label for this object
        label = labels[int(box[5])]

        # Check if this object is a stop sign
        if label == "stop-sign":
            # Get the bounding box coordinates for this stop sign
            x1, y1, x2, y2 = box[:4]

            # Calculate the width and height of the bounding box
            width = x2 - x1
            height = y2 - y1

            # Print the location and size of the stop sign
            print(f"Stop sign detected at ({x1:.1f}, {y1:.1f}) with size ({width:.1f}, {height:.1f})")

    return


# Find if anyone is subscribed to this node
def has_subscribers():
    # False if 0 connections for both - no need to run node
    return (sign_detect_pub.get_num_connections()) or (sign_size_pub.get_num_connections())


########################################################################
### Main loop:
if __name__ == "__main__":
    # Node name
    rospy.init_node("route_stop_sign_detector", anonymous=False)

    # Dynamic Reconfigure parameter server
    srv = Server(StopSignDetectConfig, dyn_rcfg_cb)  # Using common cfg file for entire project

    # Image input from topic - from launch file
    imgtopic = rospy.get_param("~imgtopic_name")
    rospy.Subscriber(imgtopic, Image, get_image, queue_size=1)

    # Topics to output detection results at
    sign_detect_topic = rospy.get_param("~stop_sign_detected_topic_name")
    sign_size_topic = rospy.get_param("~stop_sign_size_topic_name")
    sign_detect_pub = rospy.Publisher(sign_detect_topic, UInt8, queue_size=10)  # Bigger queue size???
    sign_msg = UInt8()  # Bool 0==False, !0==True
    sign_size_pub = rospy.Publisher(sign_detect_topic, UInt32, queue_size=10)  # to check for false positives???
    size_msg = UInt32()  # Detection box area (h x w in pixels)

    # Start Looping
    try:
        while not rospy.is_shutdown():
            if has_subscribers():  # At least one subscriber connected
                if not enabled:  # Startup if not enabled
                    model_path = rospy.get_param("~model_path_from_root")  # Load latest path
                    model = YOLO(model_path)  # Reload model (allows changing model while running)
                    enabled = True
            else:  # No subscribers connected
                if enabled:
                    enabled = False  # Shutdown
            rospy.spin()  # Runs callbacks
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()  # Close all windows
