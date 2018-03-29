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

resources_dir = 'resources/'
output_dir = 'output/'
images_dir = resources_dir + 'tiny-imagenet-200/train_short/'
tf_records = output_dir + 'TFRecord.record'
# images_dir = resources_dir + 'tiny-imagenet-200/train/'

flags = tf.app.flags
flags.DEFINE_string('output_path', '', output_dir + 'TFRecord')
FLAGS = flags.FLAGS

def main():
    # Build dictionary of bounding boxes using txt files from imagenet data.  Format is dict[image_filename] = [x_min, y_min, x_max, y_max]
    bbox_dict = get_image_data(images_dir)

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
                tf_example = create_tf_record(filename, jpeg_bytes, bbox_dict)
                writer.write(tf_example.SerializeToString())
    writer.close()

def create_tf_record(fn, image_data, bbox_dict):
    # TODO START: Populate the following variables from your example.
    height = 64  # Image height
    width = 64  # Image width
    filename = fn.encode()
    encoded_image_data = image_data
    image_format = 'jpeg'.encode()  # b'jpeg' or b'png'
    # print(bbox_dict)
    # print(bbox_dict[fn][1])
    xmin = bbox_dict[fn][0]  # List of normalized left x coordinates in bounding box (1 per box)
    xmax = bbox_dict[fn][2]  # List of normalized right x coordinates in bounding box (1 per box)
    ymin = bbox_dict[fn][1]  # List of normalized top y coordinates in bounding box (1 per box)
    ymax = bbox_dict[fn][3]  # List of normalized bottom y coordinates in bounding box (1 per box)
    class_str = (fn.split('_'))[0]
    classes_text = class_str.encode()  # List of string class name of bounding box (1 per box)
    classes = int(class_str[1:])  # List of integer class id of bounding box (1 per box)
    # TODO END
    feature_set = tf.train.Features(feature={'image/height':tf.train.Feature(int64_list=tf.train.Int64List(value=[height])), 'image/width':tf.train.Feature(int64_list=tf.train.Int64List(value=[width])), 'image/filename':tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])), 'image/source_id':tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])), 'image/encoded':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])), 'image/format':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])), 'image/object/bbox/xmin':tf.train.Feature(int64_list=tf.train.Int64List(value=[xmin])), 'image/object/bbox/ymin':tf.train.Feature(int64_list=tf.train.Int64List(value=[ymin])), 'image/object/bbox/xmax':tf.train.Feature(int64_list=tf.train.Int64List(value=[xmax])), 'image/object/bbox/ymax':tf.train.Feature(int64_list=tf.train.Int64List(value=[ymax])), 'image/object/class/text':tf.train.Feature(bytes_list=tf.train.BytesList(value=[classes_text])), 'image/object/class/label':tf.train.Feature(int64_list=tf.train.Int64List(value=[classes])),})
    tf_label_and_data = tf.train.Example(features=feature_set)
    return tf_label_and_data

def get_image_data(dir):
    bb_dict = {}
    for root, dirs, files in os.walk(dir):
        for filename in files:
            file_ext = (filename.split('.'))[1]
            if file_ext=='txt':
                f=open(os.path.join(root, filename),'r')
                lines=f.readlines()
                f.close()
                for line in lines:
                    image_fn = (line.split())[0]
                    bb_dict[image_fn] = [int(x) for x in (line.split())[1:]]
    return bb_dict


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
