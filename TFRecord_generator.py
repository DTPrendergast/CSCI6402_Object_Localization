#!/usr/bin/env python
import sys
import os
import csv
import copy
import tensorflow as tf

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

building_val_TFRecord = True

resources_dir = 'resources/'
output_dir = 'resources/training_data/'
images_dir = resources_dir + 'tiny-imagenet/val/'
tf_records = output_dir + 'TFRecord_val.record'
label_map_fp = resources_dir + 'tiny-imagenet/label_map.txt'
# images_dir = resources_dir + 'tiny-imagenet-200/train/'

flags = tf.app.flags
flags.DEFINE_string('output_path', '', output_dir + 'TFRecord')
FLAGS = flags.FLAGS

def main():
    # Build dictionary of bounding boxes using txt files from imagenet data.  Format is dict[image_filename] = [x_min, y_min, x_max, y_max]
    bbox_dict = get_image_data(images_dir)
    label_map_dict = get_label_data(label_map_fp)

    # Build the TFRecords
    writer = tf.python_io.TFRecordWriter(tf_records)
    for root, dirs, files in os.walk(images_dir):
        for filename in files:
            file_ext = (filename.split('.'))[1]
            if file_ext=='JPEG':
                with open(os.path.join(root, filename), 'rb') as f:
                    jpeg_bytes = f.read()
                bytes = tf.placeholder(tf.string)
                decoded_jpeg = tf.image.decode_jpeg(bytes, channels=3)
                tf_example = create_tf_record(filename, jpeg_bytes, label_map_dict, bbox_dict)
                writer.write(tf_example.SerializeToString())
    writer.close()

def create_tf_record(fn, image_data, label_map_dict, bbox_dict):
    # TODO START: Populate the following variables from your example.
    height = 64  # Image height
    width = 64  # Image width
    filename = fn.encode()
    encoded_image_data = image_data
    image_format = 'jpeg'.encode()  # b'jpeg' or b'png'
    # print(bbox_dict)
    # print(bbox_dict[fn][1])
    offset = 0
    if building_val_TFRecord:
        offset = 1
    xmins = float(bbox_dict[fn][0 + offset])/width  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = float(bbox_dict[fn][2 + offset])/width  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = float(bbox_dict[fn][1 + offset])/height  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = float(bbox_dict[fn][3 + offset])/height  # List of normalized bottom y coordinates in bounding box (1 per box)
    n_number = (fn.split('_'))[0]
    if building_val_TFRecord:
        n_number = bbox_dict[fn][0]
    classes_text = n_number.encode()  # List of string class name of bounding box (1 per box)
    classes = label_map_dict[n_number]  # List of integer class id of bounding box (1 per box)
    # TODO END
    feature_set = tf.train.Features(feature={'image/height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])), 'image/width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])), 'image/filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])), 'image/source_id':tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])), 'image/encoded':tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])), 'image/format':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])), 'image/object/bbox/xmin':tf.train.Feature(float_list=tf.train.FloatList(value=[xmins])), 'image/object/bbox/ymin':tf.train.Feature(float_list=tf.train.FloatList(value=[ymins])), 'image/object/bbox/xmax':tf.train.Feature(float_list=tf.train.FloatList(value=[xmaxs])), 'image/object/bbox/ymax':tf.train.Feature(float_list=tf.train.FloatList(value=[ymaxs])), 'image/object/class/text':tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes_text])), 'image/object/class/label':tf.train.Feature(int64_list=tf.train.Int64List(value=[classes])),})
    tf_label_and_data = tf.train.Example(features=feature_set)
    return tf_label_and_data

def get_image_data(dir):
    dict = {}
    for root, dirs, files in os.walk(dir):
        for filename in files:
            file_ext = (filename.split('.'))[1]
            if file_ext=='txt':
                f=open(os.path.join(root, filename),'r')
                lines=f.readlines()
                f.close()
                for line in lines:
                    image_fn = (line.split())[0]
                    dict[image_fn] = [x for x in (line.split())[1:]]
    return dict

def get_label_data(fn):
    data = {}
    f=open(fn,'r')
    lines=f.readlines()
    f.close()
    for line in lines:
        # n_number = (line.split())[0]
        line_list = line.split()
        n_number = line_list[0]
        id = int(line_list[1])
        data[n_number] = id
    return data


def read_csv(fn):
    data = []
    with open(fn,'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
        csvfile.close()
    del data[0]
    return data



if __name__ == '__main__':
    main()
