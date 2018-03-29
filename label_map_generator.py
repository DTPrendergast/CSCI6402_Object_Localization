#!/usr/bin/env python
import sys
import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

resources_dir = 'resources/'
output_dir = 'output/'
label_map_txt_path = resources_dir + 'tiny_imagenet/words.txt'
label_map_pbtxt_path = resources_dir + 'data/label_map.pbtxt'
# images_dir = resources_dir + 'tiny-imagenet-200/train/'

def main():
    # Build dictionary of bounding boxes using txt files from imagenet data.  Format is dict[image_filename] = [x_min, y_min, x_max, y_max]
    label_dict = get_label_data(label_map_txt_path)
    # write_pbtxt(label_dict)


def get_label_data(fn):
    data = {}
    f=open(fn,'r')
    lines=f.readlines()
    f.close()
    for line in lines:
        # n_number = (line.split())[0]
        n_number = line[0:9]
        id = int(n_number[1:])
        data[id] = line[9:]
        print(n_number, id, data[id])
    return data

# def write_pbtxt(dict):





if __name__ == '__main__':
    main()
