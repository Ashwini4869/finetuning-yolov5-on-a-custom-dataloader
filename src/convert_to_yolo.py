"""
WIDER Face to YOLO Format Converter

This script converts WIDER Face dataset annotations to YOLO format for object detection training.
It processes both training and validation datasets, converting bounding box coordinates and
creating the necessary directory structure for YOLO training.

The WIDER Face dataset uses absolute coordinates (x, y, width, height), while YOLO format
requires normalized coordinates (xcenter, ycenter, width, height) relative to image dimensions.
"""

import os
from shutil import copyfile
from PIL import Image

TRAIN_IMG_DIR = r"../Dataset/WIDER_train/images/"
VALID_IMG_DIR = r"../Dataset/WIDER_val/images/"
TRAIN_ANNOTATIONS_DIR = r"../Dataset/WIDER_train/wider_face_train_bbx_gt.txt"
VALID_ANNOTATIONS_DIR = r"../Dataset/WIDER_val/wider_face_val_bbx_gt.txt"

TRAIN_DST_DIR = r"../Dataset_YOLO/train/"
VALID_DST_DIR = r"../Dataset_YOLO/valid/"


def convert(annotations_file, src_dir, dst_dir, yolo_file):
    yolo_file = open(yolo_file, "w+")

    number_of_files = 0
    with open(annotations_file, "r") as fp:
        line = fp.readline()
        nObject = 0
        while line:
            if nObject == 0:
                oStr = fp.readline()
                nObject = int(oStr)
                line = line.strip("\r\n")
                fname, fextension = os.path.splitext(line)
                fname = fname.split("/").pop()
                width, height = Image.open(src_dir + line).size
                yolo_file.write("%s\n" % (dst_dir + fname + fextension))
                labels_file = open(dst_dir + fname + ".txt", "w+")
                copyfile(src_dir + line, dst_dir + fname + fextension)

                number_of_files += 1
                print("Total file: %d" % number_of_files, end="\r")

                if nObject == 0:
                    line = fp.readline()

            else:
                line = fp.readline()
                params = line.strip("\r\n").split(" ")
                xCenter = (float(params[0]) + float(params[2]) / 2) / width
                yCenter = (float(params[1]) + float(params[3]) / 2) / height
                w = float(params[2]) / width
                h = float(params[3]) / height
                labels_file.write("0 %.4f %.4f %.4f %.4f" % (xCenter, yCenter, w, h))
                nObject -= 1
                if nObject != 0:
                    labels_file.write("\n")

            if nObject == 0:
                labels_file.close()
                line = fp.readline()

    fp.close()
    print("Total files: %d" % number_of_files)


# Run for train and valid
print("Converting train dataset to yolo format")
convert(
    annotations_file=TRAIN_ANNOTATIONS_DIR,
    src_dir=TRAIN_IMG_DIR,
    dst_dir=TRAIN_DST_DIR,
    yolo_file="yolo_train.txt",
)
print("Converted train dataset t yolo format")
print("Converting valid dataset to yolo format")
convert(
    annotations_file=VALID_ANNOTATIONS_DIR,
    src_dir=VALID_IMG_DIR,
    dst_dir=VALID_DST_DIR,
    yolo_file="yolo_val.txt",
)
print("Converted valid dataset to yolo format")
