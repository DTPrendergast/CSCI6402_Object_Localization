# Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

# Some of the code is copied from Google's example at
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

# and some is copied from Dat Tran's example at
# https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

# but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
import matplotlib.pyplot as plt

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Name of the directory containing the object detection module we're using
MODEL_DIR = 'output/ssd_inception_v2/inference_graph'
PATH_TO_CKPT = os.path.join(MODEL_DIR,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = 'resources/training_data/label_map.pbtxt'
val_imgs_dir = 'resources/tiny-imagenet/val/'
val_imgs_fp = val_imgs_dir + 'images/'
val_img_key_fp = val_imgs_dir + 'val_annotations.txt'

# Number of classes the object detector can identify
NUM_CLASSES = 200



def main():
    val_img_dict = get_val_image_data(val_img_key_fp)


    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    num_correct = 0
    results = []
    for fn in val_img_dict:
        # val_img_dict structure:  dict[img_fn] = {'img_fn':img_fn, 'class_id':class_id, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax}

        true_n_number = val_img_dict[fn]['class_id']
        true_xmin = val_img_dict[fn]['xmin']
        true_ymin = val_img_dict[fn]['ymin']
        true_xmax = val_img_dict[fn]['xmax']
        true_ymax = val_img_dict[fn]['ymax']

        img_fp = val_imgs_fp + fn
        # Load image using OpenCV and
        # expand image dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        image = cv2.imread(img_fp)
        print(img_fp)
        # plt.imshow(image)
        # plt.show()

        image_expanded = np.expand_dims(image, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})

        # print(int(classes[0][0]))
        class_id_dict = category_index.get(classes[0][0])
        n_number = class_id_dict['name']
        # print(class_id_dict)
        bbox = boxes[0][0]
        # print(bbox)
        # print(scores[0][0])
        # print([category_index.get(i) for i in classes[0] if scores[0, i] > min_score_thresh])
        # Draw the results of the detection (aka 'visulaize the results')

        if n_number==true_n_number or true_n_number=='n03930313':
            if n_number==true_n_number:
                num_correct += 1
            xmin = int(np.round(bbox[0] * 64))
            ymin = int(np.round(bbox[1] * 64))
            xmax = int(np.round(bbox[2] * 64))
            ymax = int(np.round(bbox[3] * 64))

            true_xmin = val_img_dict[fn]['xmin']
            true_ymin = val_img_dict[fn]['ymin']
            true_xmax = val_img_dict[fn]['xmax']
            true_ymax = val_img_dict[fn]['ymax']

            # Calculate overlap
            predicted_bb = {'x1':xmin, 'x2':xmax, 'y1':ymin, 'y2':ymax}
            true_bb = {'x1':true_xmin, 'x2':true_xmax, 'y1':true_ymin, 'y2':true_ymax}
            overlap = get_overlap(predicted_bb, true_bb)
            if true_n_number!=n_number:
                overlap = 'na'

            # Record results
            row = [true_n_number, n_number, overlap]
            results.append(row)

            # Display image
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            cv2.rectangle(image, (true_xmin, true_ymin), (true_xmax, true_ymax), (0, 255, 0), 1)
            large_img = cv2.resize(image, (0,0), fx=4.0, fy=4.0)
            cv2.imshow('Object detector', large_img)
            # Press any key to close the image
            # cv2.waitKey(0)
            # save image
            cv2.imwrite('output/' + true_n_number + str(num_correct) + '-' + str(overlap) + 'percent.png', large_img)
            # Clean up
            cv2.destroyAllWindows()
            # plt.imshow(image)
            # plt.show()

    for row in results:
        print(row)


def get_overlap(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    if bb1['x1'] >= bb1['x2'] or bb1['y1'] >= bb1['y2'] or bb2['x1'] >= bb2['x2'] or bb2['y1'] >= bb2['y2']:
        return 0.0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    if iou>1.0 or iou<0.0:
        return 0.0
    return iou


def get_val_image_data(fp):
    dict = {}
    f=open(fp,'r')
    lines=f.readlines()
    f.close()
    for line in lines:
        line_list = line.split()
        img_fn = line_list[0]
        class_id = line_list[1]
        xmin = int(line_list[2])
        ymin = int(line_list[3])
        xmax = int(line_list[4])
        ymax = int(line_list[5])
        dict[img_fn] = {'img_fn':img_fn, 'class_id':class_id, 'xmin':xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax}
    return dict

if __name__ == '__main__':
    main()
