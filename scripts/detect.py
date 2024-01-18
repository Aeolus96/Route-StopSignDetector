#!/usr/bin/env python3
########################################################################
import rospy  # ROS API
from sensor_msgs.msg import Image  # Image message
from std_msgs.msg import UInt8, UInt32  # UInt8(used as bool) and UInt32(used as +ve number) message
from ultralytics import YOLO  # YOLOv8
import numpy as np  # numpy
import torch
import cv2  # OpenCV version 2.x (ROS limitation)
from cv_bridge import CvBridge, CvBridgeError  # ROS to/from OpenCV API
from dynamic_reconfigure.server import Server  # ROS Parameter server for debug
from ltu_actor_route_stop_sign_detector.cfg import StopSignDetectConfig  # packageName.cfg

########################################################################
### Global Variables:
global config_  # Dynamic reconfiguration holder
global bridge  # ROS-CV bridge
bridge = CvBridge()
global model_stop_path  # Get yolov8 stop sign model's path
global model_coco_path  # Get yolov8 default model's path
global model_stop  # Model to distinguish between stop sign categories
global model_coco  # Model to find stop sign bounding boxes
global display_size
display_size = 640  # pixel resolution for input data
global image_size
global frame_counter
frame_counter = 0

########################################################################
### Functions:


# Make a local global copy of the parameter configuration
def dyn_rcfg_cb(config, level):
    global config_
    config_ = config
    return config


# Image callback - Converts ROS Image to OpenCV Image and feeds it to the YOLO layer to then publish stop sign detection results to ROS topics
def get_image(Image):
    global frame_counter
    if config_.enable:
        frame_counter += 1
        if frame_counter > config_.frame_rate:
            frame_counter = 0
            global bridge
            try:  # convert ros_image into an opencv-compatible image
                cv_image = bridge.imgmsg_to_cv2(Image, "bgr8")
            except CvBridgeError as e:
                print(e)
            # Now cv_image is a standard OpenCV matrice

            # Resize according to dynamic reconfigure settings
            cv_image = resize_image(cv_image, size=(config_.image_resize, config_.image_resize))

            global image_size  # find the pixel resolution [h x w]
            height = cv_image.shape[0]
            width = cv_image.shape[1]
            image_size = height * width

            if config_.flip_image:  # flip the image if camera is mounted upside-down
                cv_image = cv2.flip(cv_image, 0)  # Upside down
                cv_image = cv2.flip(cv_image, 1)  # Side to side

            publish_results(find_stop_sign(cv_image))

    else:
        cv2.destroyAllWindows()  # Close all OpenCV display windows if this node is disabled temporarily
    return


# Resize cv_image to desired size with black letterbox
# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def resize_image(img, size=(display_size, display_size)):
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
    # >>> STAGE 1: Detection using two models (COCO dataset model is really good for stop signs in good lighting
    # conditions without obstructions or vandalism. The second custom trained model is for stop signs that are
    # not in the best of conditions or to detect fake stop signs)

    # Use default YOLOv8m model trained on the COCO dataset to find bounding box of the stop sign
    coco_res = model_coco(
        source=cv_image,  # Image source
        classes=11,  # only detect class name "stop sign" from COCO dataset
        device="0",  # Use GPU for inference
        stream=True,
        verbose=False,
        conf=0.5,
    )
    # Use YOLOv8m model trained on the StopSignDetection (with fake signs) dataset to find bounding box of the stop sign
    stop_res = model_stop(
        source=cv_image,  # Input image
        agnostic_nms=True,  # Prevents overlapping classes (selects class with highest confidence)
        device="0",  # Use GPU
        stream=True,
        verbose=False,
        conf=0.5,
    )
    # Convert generator output to list for easy access
    # coco_results = list(coco_results)
    # stop_results = list(stop_results)

    # >>> STAGE 2: Analyze results

    # Used to pass results to the next function:
    at_least_one_stop_sign = False  # Not confirmed yet
    biggest_bounding_box = 0.0  # Only get bounding box after confirmation
    stop_detected = False
    fake_detected = False
    coco_detected = False

    for coco_results in coco_res:
        # Show the detected images real-time for debug
        try:
            coco_results_img = resize_image(coco_results[0].plot())
        except:
            pass
        coco_boxes = coco_results[0].boxes.cpu().numpy()  # Only cast boxes to numpy
        coco_labels = coco_results[0].names  # Direct class labels access
        # {11: "stop sign"}

        for coco_idx, coco_box in enumerate(coco_boxes):  # Iterate coco bounding boxes and find the largest one
            coco_label = coco_labels[int(coco_box.cls)]  # Get the class label
            if coco_label == "stop sign":  # Check if this object is an actual stop sign
                coco_detected = True
                # Find the width and height of the bounding box
                box_width = coco_box.xywh[0][2]
                box_height = coco_box.xywh[0][3]
                area = 100 * ((box_width * box_height) / image_size)  # Percent Area
                if area > biggest_bounding_box:  # Store the largest bounding box
                    biggest_bounding_box = area
    
    for stop_results in stop_res:
        # Show the detected images real-time for debug
        try:
            stop_results_img = resize_image(stop_results[0].plot())
        except:
            pass
        stop_boxes = stop_results[0].boxes.cpu().numpy()  # Only cast boxes to numpy
        stop_labels = stop_results[0].names  # Direct class labels access
        # {0: 'stop-sign', 1: 'stop-sign-fake', 2: 'stop-sign-obstructed', 3: 'stop-sign-vandalized'}
        for stop_idx, stop_box in enumerate(stop_boxes):  # Iterate stop bounding boxes and find the largest one
            stop_label = stop_labels[int(stop_box.cls)]  # Get the class label
            if stop_label == "stop-sign" or stop_label == "stop-sign-obstructed" or stop_label == "stop-sign-vandalized":
                stop_detected = True
                # Find the width and height of the bounding box
                box_width = stop_box.xywh[0][2]
                box_height = stop_box.xywh[0][3]
                area = 100 * ((box_width * box_height) / image_size)  # Percent Area
                if area > biggest_bounding_box:  # Store the largest bounding box
                    biggest_bounding_box = area
            if stop_label == "stop-sign-fake":
                fake_detected = True

    # >>> STAGE 3: Determine the logic
    # Temporary logic for determining stop signs with two models
    # Will be incorrect if two actual signs are detected at the same time, one fake and another legitimate
    # ***Bug ignored for now as IGVC does not specify a condition like that***
    if not fake_detected:  # No fake stop signs
        if coco_detected or stop_detected:  # Either model detected a valid stop sign
            # Added a condition to sanity check if biggest bounding box is under a threshold value
            # Eg: 70% of image. If the sign is that close to the camera, the car has already crashed into the sign.
            if biggest_bounding_box < config_.box_limit:  # helps eliminate false positives for vandalized signs
                at_least_one_stop_sign = True

    # Show the detected images real-time for debug
    # coco_results_img = resize_image(coco_results[0].plot())
    # stop_results_img = resize_image(stop_results[0].plot())
    vertically_stacked_img = np.concatenate((coco_results_img, stop_results_img), axis=0)
    vertically_stacked_img = bridge.cv2_to_imgmsg(vertically_stacked_img, "bgr8")
    debug_pub.publish(vertically_stacked_img)

    return at_least_one_stop_sign, int(biggest_bounding_box * 100)  # bool, int (Percent * 100, 43% = 4300)


# Simple ROS message publisher
def publish_results(results):
    (detected, area) = results
    sign_msg = UInt8()  # Bool 0==False, !0==True
    sign_msg.data = detected
    sign_detect_pub.publish(sign_msg)

    size_msg = UInt32()  # Detection box area (Percent Area * 100)
    size_msg.data = area
    sign_size_pub.publish(size_msg)
    return

########################################################################
### Main loop:
if __name__ == "__main__":
    # Node name
    rospy.init_node("route_stop_sign_detector", anonymous=False)

    # Load the YOLO model at startup
    model_stop_path = rospy.get_param("~model_stop_path_from_root")  # Load latest path
    model_stop = YOLO(model_stop_path)
    model_coco_path = rospy.get_param("~model_coco_path_from_root")  # Load latest path
    model_coco = YOLO(model_coco_path)

    # Dynamic Reconfigure parameter server
    srv = Server(StopSignDetectConfig, dyn_rcfg_cb)  # Using common cfg file for entire project

    # Image input from topic - from launch file
    imgtopic = rospy.get_param("~imgtopic_name")
    rospy.Subscriber(imgtopic, Image, get_image, queue_size=1)

    # Topics to output detection results at
    sign_detect_topic = rospy.get_param("~stop_sign_detected_topic_name")
    sign_size_topic = rospy.get_param("~stop_sign_size_topic_name")
    debug_topic = rospy.get_param("~stop_sign_debug_topic_name")
    sign_detect_pub = rospy.Publisher(sign_detect_topic, UInt8, queue_size=10)  # Bigger queue size???
    sign_size_pub = rospy.Publisher(sign_size_topic, UInt32, queue_size=10)  # to check for false positives???
    debug_pub = rospy.Publisher(debug_topic, Image, queue_size=1) #to display image as a topic
    
    # Start Looping
    try:
        while not rospy.is_shutdown():
            torch.cuda.empty_cache()
            rospy.spin()  # Runs callbacks
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()  # Close all windows
