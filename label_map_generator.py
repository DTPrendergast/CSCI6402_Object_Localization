#!/usr/bin/env python
import sys
import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

resources_dir = 'resources/tiny-imagenet/'
output_dir = 'resources/data/'
label_map_txt_path = resources_dir + 'words.txt'
label_map_pbtxt_path = output_dir + 'label_map.pbtxt'
# images_dir = resources_dir + 'tiny-imagenet-200/train/'

def main():
    # Build dictionary of bounding boxes using txt files from imagenet data.  Format is dict[image_filename] = [x_min, y_min, x_max, y_max]
    label_dict = get_label_data(label_map_txt_path)
    write_pbtxt(label_dict, label_map_pbtxt_path)


def get_label_data(fn):
    data = {}
    f=open(fn,'r')
    lines=f.readlines()
    f.close()
    for line in lines:
        # n_number = (line.split())[0]
        n_number = line[0:9]
        id = int(n_number[1:])
        data[id] = line[9:].strip()
    return data

def write_pbtxt(dict, fn):
    with open(fn, 'w') as f:
        for key in dict:
            f.write("item {\n")
            f.write("  id: " + str(key) + "\n")
            f.write("  name: " + '"' + dict[key] + '"' + "\n")
            f.write("}\n")

if __name__ == '__main__':
    main()
