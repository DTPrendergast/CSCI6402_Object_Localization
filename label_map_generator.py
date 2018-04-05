#!/usr/bin/env python
import sys
import os

# Set working directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

resources_dir = 'resources/tiny-imagenet/'
output_dir = 'resources/training_data/'
label_map_txt_path = resources_dir + 'full_label_map.txt'
new_label_map_txt_path = resources_dir + 'label_map.txt'
label_map_pbtxt_path = output_dir + 'label_map.pbtxt'
train_images_root = resources_dir + 'train/'
# images_dir = resources_dir + 'tiny-imagenet-200/train/'

def main():
    # Build dictionary of bounding boxes using txt files from imagenet data.  Format is dict[image_filename] = [x_min, y_min, x_max, y_max]
    raw_label_dict = get_label_data(label_map_txt_path)

    labels = {}
    i = 1
    for root, dirs, files in os.walk(train_images_root):
        folder_name = str(root).split('/')[-1]
        # print(folder_name)
        if folder_name!='images' and folder_name!='':
            text = raw_label_dict[folder_name][1]
            labels[i] = (folder_name, text)
            i += 1

    write_pbtxt(labels, label_map_pbtxt_path)
    write_txt(labels, new_label_map_txt_path)


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
        text = line_list[2]
        print('         ', n_number, id, text)
        data[n_number] = (id, text)
    return data

def write_pbtxt(dict, fn):
    with open(fn, 'w') as f:
        for key in dict:
            f.write("item {\n")
            f.write("  id: " + str(key) + "\n")
            f.write("  name: " + "'" + dict[key][0] + "'" + "\n")
            f.write("}\n")

def write_txt(dict, fn):
    with open(fn, 'w') as f:
        for key in dict:
            print(dict[key][0] + ' ' + str(key) + ' ' + dict[key][1] + '\n')
            f.write(dict[key][0] + ' ' + str(key) + ' ' + dict[key][1] + '\n')

if __name__ == '__main__':
    main()
