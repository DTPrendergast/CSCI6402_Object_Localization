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
images_dir = resources_dir + 'tiny-imagenet-200/train_short/'
# images_dir = resources_dir + 'tiny-imagenet-200/train/'



def main():

    for subdirs, dirs, files in os.walk(images_dir):
        print(subdirs, type(subdirs))
        for file in files:
            filename = copy.copy(file)
            file_ext = (filename.split('.'))[1]
            if file_ext=='txt':
                print(file, type(file))
                f=open(file,'r')
                lines=f.readlines()
                f.close()
                print(type(lines))
                print(len(lines))
                print(type(lines[0]))



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
