# needs pip install: tensorflow, opencv, nltk, dill
import colorsys
import copy
import csv
import cv2
import glob
import json
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pickle
import pykociemba.cubiecube as cubiecube
import pykociemba.facecube as facecube
import random
import shutil

from sklearn import tree
from sklearn.tree import export_text
from joblib import load

import tensorflow as tf
from tensorflow.keras.models import load_model

from mpl_toolkits.mplot3d import Axes3D

class CubeStatus:
    """
    Read cube status from images
    """

    def __init__(self):
        # rectangles represented by x, y, width and height
        self.polygons = {
            "U1":  (888, 248,60,33), "U2": (1025,274,46,37), "U3": (1169,313,53, 37), "U4":(765,283,62,35),
            "U5":(907,327,58,37), "U6":(1053,371,65,42), "U7":(607,339,56,42), "U8":(744,382,60,42), "U9":(904,434,70,49),
            "L1":(547,436,49,47), "L2":(670,501,51,60), "L3":(800,564,56,60), "L4":(594,607,46,51),
            "L5":(700,679,46,58), "L6":(826,765,49,60), "L7":(619,744,39,46), "L8":(719,821,42,51), "L9":(832,913,47,53),
            "R1":(995,587,49,60), "R2":(1146,501,51,62), "R3":(1269,420,44,58), "R4":(986,782,51,58),
            "R5":(1141,686,42,53), "R6":(1243,589,42,51), "R7":(992,939,44,53), "R8":(1124,823,42,47), "R9":(1222,735,40,46)
        }

        # Map labels in the first picture to final label in cube status.      
        self.pic2_label_mapping = {
                'U1': 'U1', 'U2': 'U2', 'U3': 'U3', 'U4': 'U4', 'U5': 'U5', 'U6': 'U6', 'U7': 'U7', 'U8': 'U8', 'U9': 'U9',
                'L1': 'F1', 'L2': 'F2', 'L3': 'F3', 'L4': 'F4', 'L5': 'F5', 'L6': 'F6', 'L7': 'F7', 'L8': 'F8', 'L9': 'F9',
                'R1': 'R1', 'R2': 'R2', 'R3': 'R3', 'R4': 'R4', 'R5': 'R5', 'R6': 'R6', 'R7': 'R7', 'R8': 'R8', 'R9': 'R9',
                }
        # Map labels in the second picture to final label in cube status.      
        self.pic1_label_mapping = {
                'U1': 'D3', 'U2': 'D6', 'U3': 'D9', 'U4': 'D2', 'U5': 'D5', 'U6': 'D8', 'U7': 'D1', 'U8': 'D4', 'U9': 'D7',
                'L1': 'L9', 'L2': 'L8', 'L3': 'L7', 'L4': 'L6', 'L5': 'L5', 'L6': 'L4', 'L7': 'L3', 'L8': 'L2', 'L9': 'L1',
                'R1': 'B9', 'R2': 'B8', 'R3': 'B7', 'R4': 'B6', 'R5': 'B5', 'R6': 'B4', 'R7': 'B3', 'R8': 'B2', 'R9': 'B1',
                }
        # The order of sides to output cube status
        self.side_order = ['U', 'R', 'F', 'D', 'L', 'B']
        self.colors = ['B', 'G', 'O', 'R', 'W', 'Y']
        self.model = load_model('./color_detection-v4-7.h5', compile = False)
        self.decision_tree = load('./decision_tree-v4-7.joblib')

        # For display status
        self.STICKER_AREA_TILE_SIZE = 30
        self.STICKER_AREA_TILE_GAP = 4
        self.SIDE_AREA_GAP = 20
        self.SIDE_AREA_SIZE = self.STICKER_AREA_TILE_SIZE * 3 + self.STICKER_AREA_TILE_GAP * 4
        self.SIDE_AREA_POSITION = {'U': (1, 0), 'L': (0, 1), 'F': (1, 1), 'R': (2, 1), 'B': (3, 1), 'D': (1, 2)}
        self.color_palette = {
            'R'   : (255, 0, 0),
            'O': (255, 165, 0),
            'B'  : (0, 0, 255),
            'G' : (0, 255, 0),
            'W' : (255, 255, 255),
            'Y': (255, 255, 0)
        }

        # Side to color mapping
        self.side_to_color = {}

        # image width and height for model input
        self.img_width = 96
        self.img_height = 96

        self.cam = cv2.VideoCapture(0)

        self.first_pic = ''
        self.first_pic_taken = False
        self.second_pic = ''
        self.second_pic_taken = False

    def extract_regions(self, img_files, out_path, true_label_file):
        """
        Extract regions from images and store them as jpg in specified output path.
        The true_label_file contains true labels for each region. The output file name is in format
        ture_label.original_file_name-region.jpg.
        """
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        true_label_by_file = {}
        if true_label_file != '':
            true_label_by_file = json.loads(open(true_label_file).read())
            print(true_label_by_file)
        all_true_labels = ''
        all_predicted_labels = ''
        count_per_true_label = {}
        for img_path in img_files:
            print(img_path)
            img = cv2.imread(img_path)
            idx = 0
            true_label = true_label_by_file[os.path.basename(img_path)]
            all_true_labels += true_label

            for side in ['U', 'L', 'R']:
                for num in range(1, 10):
                    region = side + str(num)
                    (x, y, width, height) = self.polygons[region]
                    rect = img[round(y) : round(y) + round(height), round(x) : round(x) + round(width)]
                    fname = "{0}.{1}-{2}.jpg".format(true_label[idx],
                            os.path.basename(img_path).split('.')[0], region)
                    count_per_true_label[true_label[idx]] = count_per_true_label.get(true_label[idx], 0) + 1
                    cv2.imwrite(os.path.join(out_path, fname), rect)
                    idx += 1

        print(count_per_true_label)

    def capture_pictures(self, repeat_times=1, output_path=''):
        """
        Capture pictures that will be used for color detection.
        The optional input for repeate_times and output_path is for capturing pictures to
        prepare training data. It will repeate repeat_times, each time two pictures and
        store the pictures in the output_path if specified.
        After the first picture, you need to flip the cube to show the other three sides
        to take the second picture.
        """
        print('Press space on the video window to take picture!')
        repeated = 0
        while True:
            _, frame = self.cam.read()
            key = cv2.waitKey(10) & 0xff

            # Quit on escape.
            if key == 27:
                break

            if key == 32:
                if not self.first_pic_taken:
                    print('First picture taken')
                    self.first_pic_taken = True
                    if output_path != '':
                        cv2.imwrite(os.path.join(output_path, '{}-1.jpg'.format(repeated)), frame)
                        print(os.path.join(output_path, '{}-1.jpg'.format(repeated)))
                    self.first_pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    print('Second picture taken')
                    self.second_pic_taken = True
                    if output_path != '':
                        cv2.imwrite(os.path.join(output_path, '{}-2.jpg'.format(repeated)), frame)
                    self.second_pic = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.first_pic_taken and self.second_pic_taken:
                    repeated += 1
                    if repeated < repeat_times:
                        self.first_pic_taken = False
                        self.second_pic_taken = False
                    else:
                        break

            cv2.imshow('Rubik cube', frame)
        self.cam.release()
        cv2.destroyAllWindows()

    def predict_colors_with_cnn(self, img):
        predicts = {}
        for label in self.polygons:
          (x, y, width, height) = self.polygons[label]
          rect = img[round(y):round(y)+round(height), round(x):round(x)+round(width)]
          rect = cv2.resize(rect, (self.img_width, self.img_height))
          resized = tf.reshape(rect, [-1, 96, 96, 3])
          predict = self.model.predict(resized)
          predicts[label] = np.argmax(predict)
        return predicts

    def get_dominant_hsv(self, img):
        average = img.mean(axis=0).mean(axis=0)
        pixels = np.float32(img.reshape(-1, 3))

        n_colors = 1
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        # dominant BGR
        (b, g, r) = palette[np.argmax(counts)]

        # convert BGR to HSV
        (h, s, v) = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        return (h * 360, s * 100, v * 100)

    def predict_colors_with_decision_tree(self, img):
        inputs = []
        labels = []
        for label in self.polygons:
          labels.append(label)
          (x, y, width, height) = self.polygons[label]
          rect = img[round(y):round(y)+round(height), round(x):round(x)+round(width)]
          (h, s, v) = self.get_dominant_hsv(rect)
          inputs.append([h, s, v])
        predicts = self.decision_tree.predict(inputs)
        output = {}
        for idx, label in enumerate(labels):
          output[label] = predicts[idx]
        return output

    def convert_to_status_input(self, predicts, label_mapping):
        """
        Convert regions in one picture to regions in cube.
        """
        label_to_color = {}
        for l in predicts:
          color = self.colors[predicts[l]]
          label = label_mapping[l]
          label_to_color[label] = color
        return label_to_color

    def generate_status_str(self, label_to_color):
        """
        Generate status string for the whole cube.
        """
        labels =  list(label_to_color.keys())
        side_to_color = {'U': label_to_color['U5'], 'F': label_to_color['F5'], 'R': label_to_color['R5'],
                         'L': label_to_color['L5'], 'B': label_to_color['B5'], 'D': label_to_color['D5']}

        # If we don't have one color for each side, ask input to fix them
        color_to_side = {v: k for k, v in side_to_color.items()}
        while len(color_to_side.keys()) != 6:
            print(side_to_color)
            side = input('Color for one of the sides is wrongly detected, which side is it? (U, F, R, L, B, D)\n')
            while not side in ['U', 'F', 'R', 'L', 'B', 'D']:
              side = input('Please pick side from (U, F, R, L, B, D).\n')
            color = input('What is the true color for this side? (R, O, Y, B, G, W)\n')
            while not color in ['R', 'O', 'Y', 'B', 'G', 'W']:
              color = input('Please pick color from (R, O, Y, B, G, W).\n')
            side_to_color[side] = color
            label_to_color[side + '5'] = color
            color_to_side = {v: k for k, v in side_to_color.items()}

        status_string = ''
        for side in self.side_order:
          for i in range(1, 10):
            key = '{}{}'.format(side, i)
            color = label_to_color[key]
            out_side = color_to_side[color]
            status_string += out_side
        return (status_string, side_to_color)

    def draw_stickers(self, img, stickers, offset, side_to_color):
        for idx, scolor in enumerate(stickers):
          row = math.floor(idx / 3)
          col = idx % 3
          sticker_offset_x = col * self.STICKER_AREA_TILE_SIZE + (col + 1) * self.STICKER_AREA_TILE_GAP
          sticker_offset_y = row * self.STICKER_AREA_TILE_SIZE + (row + 1) * self.STICKER_AREA_TILE_GAP
          color = side_to_color[scolor]
          (r, g, b) = self.color_palette[color]
          cv2.rectangle(img, (offset[0] + sticker_offset_x, offset[1] + sticker_offset_y),
                        (offset[0] + sticker_offset_x + self.STICKER_AREA_TILE_SIZE,
                        offset[1] + sticker_offset_y + self.STICKER_AREA_TILE_SIZE),
                        (b, g, r), -1)

    def validate_color_count(self, status_list, side_to_color):
        color_count = {}
        has_none_9 = False
        for c in set(status_list):
          count = status_list.count(c)
          color_count[side_to_color[c]] = count
          if count != 9:
            has_none_9 = True
        return(color_count, has_none_9)

    def display_status(self, status_list, side_to_color):
        # (width, height)
        img_width = self.SIDE_AREA_SIZE * 4 + self.SIDE_AREA_GAP * 5
        img_height = self.SIDE_AREA_SIZE * 3 + self.SIDE_AREA_GAP * 4
        img = np.zeros((img_height, img_width, 3), np.uint8)

        for idx, side in enumerate(self.side_order):
          (x, y) = self.SIDE_AREA_POSITION[side]
          offset = (x * self.SIDE_AREA_SIZE + (x + 1) * self.SIDE_AREA_GAP,
                    y * self.SIDE_AREA_SIZE + (y + 1) * self.SIDE_AREA_GAP)
          stickers = status_list[idx * 9: (idx + 1) * 9]
          self.draw_stickers(img, stickers, offset, side_to_color)
        return img

    def display_and_validate_status(self, status_string, side_to_color):
        """
        Display status and validate count for each color, if invalid ask input to fix.
        """
        print(side_to_color)
        # Build color_to_side
        color_to_side = {}
        for s in side_to_color.keys():
            color_to_side[side_to_color[s]] = s

        # count colors
        status_list = list(status_string)
        color_count, has_none_9 = self.validate_color_count(status_list, side_to_color)
        print(color_count)
        if has_none_9:
          print("Some recognized colors need to be fixed.")

        img = self.display_status(status_list, side_to_color)

        while True:
          cv2.imshow('Recognized', img)
          key = cv2.waitKey(10) & 0xff

          if has_none_9:
            # Get input to fix colors
            sq, color = input('Enter square and color to fix in format side:color.\n').split(':')
            while not (sq[0] in self.side_order and int(sq[1]) in range(1, 10) and color in self.colors):
              sq, color = input('Invalid side or color, please input again.\n').split(':')
            sq_idx = self.side_order.index(sq[0]) * 9 + int(sq[1]) - 1
            low_idx = math.floor(sq_idx / 9) * 9
            high_idx = (math.floor(sq_idx / 9) + 1) * 9
            status_list[sq_idx] = color_to_side[color]

            # Update the color in the image
            (x, y) = self.SIDE_AREA_POSITION[sq[0]]
            offset = (x * self.SIDE_AREA_SIZE + (x + 1) * self.SIDE_AREA_GAP,
                      y * self.SIDE_AREA_SIZE + (y + 1) * self.SIDE_AREA_GAP)
            stickers = status_list[low_idx : high_idx]
            self.draw_stickers(img, stickers, offset, side_to_color)

            color_count, has_none_9 = self.validate_color_count(status_list, side_to_color)
            print(color_count)

          # Quit on escape.
          if key == 27 and not has_none_9:
              break
        cv2.destroyAllWindows()
        return ''.join(status_list)


    def print_predicts(self, predicts):
        for label in predicts:
            color = self.colors[predicts[label]]
            print(label, color)

    def detect_status(self):
        # Capture pictures
        self.capture_pictures(1, '/Users/wangyu/Downloads')
        #self.capture_pictures()

        #self.first_pic = cv2.cvtColor(cv2.imread('/Users/wangyu/Downloads/0-1.jpg'), cv2.COLOR_BGR2RGB)
        #self.second_pic = cv2.cvtColor(cv2.imread('/Users/wangyu/Downloads/0-2.jpg'), cv2.COLOR_BGR2RGB)
        #self.first_pic = cv2.imread('/Users/wangyu/Downloads/0-1.jpg')
        #self.second_pic = cv2.imread('/Users/wangyu/Downloads/0-2.jpg')

        # Predict colors for captured pictures
        first_pic_predicts = self.predict_colors_with_cnn(self.first_pic)
        second_pic_predicts = self.predict_colors_with_cnn(self.second_pic)

        #first_pic_predicts = self.predict_colors_with_decision_tree(self.first_pic)
        #second_pic_predicts = self.predict_colors_with_decision_tree(self.second_pic)
        #self.print_predicts(first_pic_predicts)
        #self.print_predicts(second_pic_predicts)

        # Map per picture label to merged label
        label_to_color = self.convert_to_status_input(second_pic_predicts, self.pic2_label_mapping)
        label_to_color.update(self.convert_to_status_input(first_pic_predicts, self.pic1_label_mapping))

        # Generate status string and diaplsy
        (status_string, side_to_color) = self.generate_status_str(label_to_color)
        print(status_string)
        validated_status = self.display_and_validate_status(status_string, side_to_color)
        
        return (validated_status, side_to_color)

    def change_status(self, input_status, moves):
        """
        Given input status and the moves to apply, output the new status.
        """
        face_cube = facecube.FaceCube(input_status)
        cubie_cube = face_cube.toCubieCube()
        for move in moves:
            print(move)
            cube_to_multiply = copy.deepcopy(cubiecube.moveCube[self.side_order.index(move[0])])
            if len(move) == 2:
                if move[1] == '2':
                    cube_to_multiply.multiply(cube_to_multiply)
                if move[1] == "'":
                    one_move = copy.deepcopy(cube_to_multiply)
                    cube_to_multiply.multiply(cube_to_multiply)
                    cube_to_multiply.multiply(one_move)
            cubie_cube.multiply(cube_to_multiply)
        return cubie_cube.toStatusString()

    def show_polygons(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for label in self.polygons:
          (x, y, width, height) = self.polygons[label]
          cv2.rectangle(img, (round(x), round(y)), (round(x) + round(width), round(y) + round(height)), (255, 0, 0))
        plt.imshow(img)
        plt.show()

def split_training_testing(path):
    training_path = os.path.join(path, 'training')
    testing_path = os.path.join(path, 'testing')
    print(training_path)
    if os.path.exists(training_path):
        shutil.rmtree(training_path)
    os.makedirs(training_path)
    if os.path.exists(testing_path):
        shutil.rmtree(testing_path)
    os.makedirs(testing_path)

    for f in glob.glob(os.path.join(path, '*.jpg')):
      basename = os.path.basename(f)
      if random.randint(1, 10) <= 2:
          shutil.move(f, os.path.join(testing_path, basename))
      else:
          shutil.move(f, os.path.join(training_path, basename))


def main():
    cube_status = CubeStatus()
    # capture pictures
    #cube_status.capture_pictures(10, '/Users/wangyu/Pictures/v7-day-sidelight')

    # detect status
    #cube_status.detect_status()

    # Extract regions and output training/testing data.
    #flist = glob.glob('/Users/wangyu/Pictures/v4-7/*.jpg')
    #cube_status.extract_regions(flist, '/Users/wangyu/Pictures/v4-7/extracted', '/Users/wangyu/Pictures/v4-7/labels')
    #split_training_testing('/Users/wangyu/Pictures/v4-7/extracted')

    # Show polygons
    #cube_status.show_polygons(os.path.join('/Users/wangyu/Downloads', '0-1.jpg'))

    # Test change_status
    input_status = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
    while True:
      print('Current status: ' + input_status)
      moves = input('Moves:\n').split()
      print(input_status)
      print(moves)
      input_status = cube_status.change_status(input_status, moves)

if __name__ == "__main__":
    main()
