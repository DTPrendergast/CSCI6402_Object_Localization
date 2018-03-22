#!/usr/bin/env python
import sys
import os
import csv
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
        print(subdir, type(subdir))
        for file in files:
            f=open(file,'r')
            lines=f.readlines()
            f.close()
            






if __name__ == '__main__':
    main()
