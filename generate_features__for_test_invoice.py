import cv2
import json
import math
import numpy as np
import pandas as pd
import os
# import csvx
import csv
import glob
from config import paths
print(paths)
from config.paths import DATASET_PATH,IMAGE_PATH,MODEL_PATH,WORD_IMAGE_PATH,WORD_FEATURES_PATH

from common.todo_fix_this_later import Constants


def col(val):
    col_dict={
        (0, 2): (255, 0, 0),
        (2, 4): (255, 64, 0),
        (4, 6): (255, 128, 0),
        (6, 8): (255, 191, 0),
        (8, 10): (255, 255, 0),
        (10, 13): (191, 255, 0),
        (13, 18): (128, 255, 0),
        (18, 121): (64, 255, 0),
        (121, 135): (0, 255, 0),
        (135, 151): (0, 255, 64),
        (151, 165): (0, 255, 128),
        (165, 181): (0, 255, 191),
        (181, 195): (0, 255, 255),
        (195, 211): (0, 191, 255),
        (211, 225): (0, 128, 255),
        (225, 241): (0, 64, 255),
        (241, 256): (0, 0, 255)
    }
    # return (255,0,0)
    for k,v in col_dict.items():
        if val >= k[0] and val <k[1]:
            return v

def heatmap(image,data_dict,op_folder):
    top_distance=[]
    bottom_distance=[]
    left_distance=[]
    right_distance=[]
    area=[]
    top_left_aligned_count=[]
    continous_top_left_aligned_count=[]
    bottom_left_aligned_count=[]
    continous_bottom_left_aligned_count=[]
    top_right_aligned_count=[]
    continous_top_right_aligned_count=[]
    bottom_right_aligned_count=[]
    continous_bottom_right_aligned_count=[]
    top_aligned_count=[]
    top_continous_aligned_count=[]
    bottom_aligned_count=[]
    bottom_continous_aligned_count=[]
    left_aligned_count=[]
    left_continous_aligned_count=[]
    right_aligned_count=[]
    right_continous_aligned_count=[]
    fully_aligned_count=[]
    fully_continous_aligned_count=[]
    left_alignment_score_array=[]
    right_alignment_score_array=[]
    aggregate_alignment_score_array=[]
    for k,v in data_dict.items():
        top_distance.append(v["dist"]["top"])
        bottom_distance.append(v["dist"]["bottom"])
        left_distance.append(v["dist"]["left"])
        right_distance.append(v["dist"]["right"])
        area.append(v["dist"]["area"])
        top_left_aligned_count.append(v["top_left_aligned"])
        continous_top_left_aligned_count.append(v["continous_top_left_aligned"])
        bottom_left_aligned_count.append(v["bottom_left_aligned"])
        continous_bottom_left_aligned_count.append(v["continous_bottom_left_aligned"])
        top_right_aligned_count.append(v["top_right_aligned"])
        continous_top_right_aligned_count.append(v["continous_top_right_aligned"])
        bottom_right_aligned_count.append(v["bottom_right_aligned"])
        continous_bottom_right_aligned_count.append(v["continous_bottom_right_aligned"])
        top_aligned_count.append(v["top_aligned"])
        top_continous_aligned_count.append(v["top_continous_aligned"])
        bottom_aligned_count.append(v["bottom_aligned"])
        bottom_continous_aligned_count.append(v["bottom_continous_aligned"])

        left_aligned_count.append(v["left_aligned"])
        left_continous_aligned_count.append(v["left_continous_aligned"])
        right_aligned_count.append(v["right_aligned"])
        right_continous_aligned_count.append(v["right_continous_aligned"])
        fully_aligned_count.append(v["fully_aligned"])
        fully_continous_aligned_count.append(v["fully_continous_aligned"])

        left_alignment_score_array.append(v["left_alignment_score"])
        right_alignment_score_array.append(v["right_alignment_score"])
        aggregate_alignment_score_array.append(v["aggregate_alignment_score"])

    top_distance_image=image.copy()
    bottom_distance_image = image.copy()
    left_distance_image = image.copy()
    right_distance_image = image.copy()
    area_image = image.copy()
    top_left_aligned_image=image.copy()
    continous_top_left_aligned_image=image.copy()
    bottom_left_aligned_image=image.copy()
    continous_bottom_left_aligned_image=image.copy()
    top_right_aligned_image=image.copy()
    continous_top_right_aligned_image=image.copy()
    bottom_right_aligned_image=image.copy()
    continous_bottom_right_aligned_image=image.copy()
    top_aligned_image=image.copy()
    top_continous_aligned_image=image.copy()
    bottom_aligned_image=image.copy()
    bottom_continous_aligned_image=image.copy()
    left_aligned_image=image.copy()
    left_continous_aligned_image=image.copy()
    right_aligned_image=image.copy()
    right_continous_aligned_image=image.copy()
    fully_aligned_image=image.copy()
    fully_continous_aligned_image=image.copy()
    left_alignment_score_image=image.copy()
    right_alignment_score_image=image.copy()
    aggregate_alignment_score_image=image.copy()

    for k, v in data_dict.items():
        area_val=((v["dist"]["area"]-min(area))/((max(area)-min(area))))*255
        cv2.rectangle(area_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]), col(area_val), thickness=cv2.FILLED)

        top_distance_val = ((v["dist"]["top"] - min(top_distance)) / max(top_distance)) * 255
        cv2.rectangle(top_distance_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]), col(top_distance_val),
                      thickness=cv2.FILLED)

        bottom_distance_val = ((v["dist"]["bottom"] - min(bottom_distance)) / max(bottom_distance)) * 255
        cv2.rectangle(bottom_distance_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]), col(bottom_distance_val),
                      thickness=cv2.FILLED)

        left_distance_val = ((v["dist"]["left"] - min(left_distance)) / max(left_distance)) * 255
        cv2.rectangle(left_distance_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]), col(left_distance_val),
                      thickness=cv2.FILLED)

        right_distance_val = ((v["dist"]["right"] - min(right_distance)) / max(right_distance)) * 255
        cv2.rectangle(right_distance_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]), col(right_distance_val),
                      thickness=cv2.FILLED)

        # ************************************************

        top_left_aligned_val = ((v["top_left_aligned"] - min(top_left_aligned_count)) / max(top_left_aligned_count)) * 255
        cv2.rectangle(top_left_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(top_left_aligned_val),
                      thickness=cv2.FILLED)

        continous_top_left_aligned_val = ((v["continous_top_left_aligned"] - min(continous_top_left_aligned_count)) / max(continous_top_left_aligned_count)) * 255
        cv2.rectangle(continous_top_left_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(continous_top_left_aligned_val),
                      thickness=cv2.FILLED)

        bottom_left_aligned_val = ((v["bottom_left_aligned"] - min(bottom_left_aligned_count)) / max(bottom_left_aligned_count)) * 255
        cv2.rectangle(bottom_left_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(bottom_left_aligned_val),
                      thickness=cv2.FILLED)

        continous_bottom_left_aligned_val = ((v["continous_bottom_left_aligned"] - min(continous_bottom_left_aligned_count)) / max(continous_bottom_left_aligned_count)) * 255
        cv2.rectangle(continous_bottom_left_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(continous_bottom_left_aligned_val),
                      thickness=cv2.FILLED)

        top_right_aligned_val = ((v["top_right_aligned"] - min(top_right_aligned_count)) / max(top_right_aligned_count)) * 255
        cv2.rectangle(top_right_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(top_right_aligned_val),
                      thickness=cv2.FILLED)

        continous_top_right_aligned_val = ((v["continous_top_right_aligned"] - min(continous_top_right_aligned_count)) / max(continous_top_right_aligned_count)) * 255
        cv2.rectangle(continous_top_right_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(continous_top_right_aligned_val),
                      thickness=cv2.FILLED)

        bottom_right_aligned_val = ((v["bottom_right_aligned"] - min(bottom_right_aligned_count)) / max(bottom_right_aligned_count)) * 255
        cv2.rectangle(bottom_right_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(bottom_right_aligned_val),
                      thickness=cv2.FILLED)

        continous_bottom_right_aligned_val = ((v["continous_bottom_right_aligned"] - min(continous_bottom_right_aligned_count)) / max(continous_bottom_right_aligned_count)) * 255
        cv2.rectangle(continous_bottom_right_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(continous_bottom_right_aligned_val),
                      thickness=cv2.FILLED)

        top_aligned_val = ((v["top_aligned"] - min(top_aligned_count)) / max(top_aligned_count)) * 255
        cv2.rectangle(top_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(top_aligned_val),
                      thickness=cv2.FILLED)

        top_continous_aligned_val = ((v["top_continous_aligned"] - min(top_continous_aligned_count)) / max(top_continous_aligned_count)) * 255
        cv2.rectangle(top_continous_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(top_continous_aligned_val),
                      thickness=cv2.FILLED)

        bottom_aligned_val = ((v["bottom_aligned"] - min(bottom_aligned_count)) / max(bottom_aligned_count)) * 255
        cv2.rectangle(bottom_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(bottom_aligned_val),
                      thickness=cv2.FILLED)

        bottom_continous_aligned_val = ((v["bottom_continous_aligned"] - min(bottom_continous_aligned_count)) / max(bottom_continous_aligned_count)) * 255
        cv2.rectangle(bottom_continous_aligned_image, (v["coords"][1], v["coords"][0]), (v["coords"][3], v["coords"][2]),
                      col(bottom_continous_aligned_val),
                      thickness=cv2.FILLED)

        # *******************************************

        left_aligned_val = ((v["left_aligned"] - min(left_aligned_count)) / max(left_aligned_count)) * 255
        cv2.rectangle(left_aligned_image, (v["coords"][1], v["coords"][0]),(v["coords"][3], v["coords"][2]),
                      col(left_aligned_val),
                      thickness=cv2.FILLED)

        left_continous_aligned_val = ((v["left_continous_aligned"] - min(left_continous_aligned_count)) / max(left_continous_aligned_count)) * 255
        cv2.rectangle(left_continous_aligned_image, (v["coords"][1], v["coords"][0]),(v["coords"][3], v["coords"][2]),
                      col(left_continous_aligned_val),
                      thickness=cv2.FILLED)

        right_aligned_val = ((v["right_aligned"] - min(right_aligned_count)) / max(right_aligned_count)) * 255
        cv2.rectangle(right_aligned_image, (v["coords"][1], v["coords"][0]),(v["coords"][3], v["coords"][2]),
                      col(right_aligned_val),
                      thickness=cv2.FILLED)

        right_continous_aligned_val = ((v["right_continous_aligned"] - min(right_continous_aligned_count)) / max(right_continous_aligned_count)) * 255
        cv2.rectangle(right_continous_aligned_image, (v["coords"][1], v["coords"][0]),(v["coords"][3], v["coords"][2]),
                      col(right_continous_aligned_val),
                      thickness=cv2.FILLED)

        fully_aligned_val = ((v["fully_aligned"] - min(fully_aligned_count)) / max(fully_aligned_count)) * 255
        cv2.rectangle(fully_aligned_image, (v["coords"][1], v["coords"][0]),(v["coords"][3], v["coords"][2]),
                      col(fully_aligned_val),
                      thickness=cv2.FILLED)

        fully_continous_aligned_val = ((v["fully_continous_aligned"] - min(fully_continous_aligned_count)) / max(fully_continous_aligned_count)) * 255
        cv2.rectangle(fully_continous_aligned_image, (v["coords"][1], v["coords"][0]),(v["coords"][3], v["coords"][2]),
                      col(fully_continous_aligned_val),
                      thickness=cv2.FILLED)

        # *******************************************

        left_alignment_score_val = ((v["left_alignment_score"] - min(left_alignment_score_array)) / max(left_alignment_score_array)) * 255
        cv2.rectangle(left_alignment_score_image, (v["coords"][1], v["coords"][0]),(v["coords"][3], v["coords"][2]),
                      col(left_alignment_score_val),
                      thickness=cv2.FILLED)

        right_alignment_score_val = ((v["right_alignment_score"] - min(right_alignment_score_array)) / max(right_alignment_score_array)) * 255
        cv2.rectangle(right_alignment_score_image, (v["coords"][1], v["coords"][0]),(v["coords"][3], v["coords"][2]),
                      col(right_alignment_score_val),
                      thickness=cv2.FILLED)

        aggregate_alignment_score_val = ((v["aggregate_alignment_score"] - min(aggregate_alignment_score_array)) / max(aggregate_alignment_score_array)) * 255
        cv2.rectangle(aggregate_alignment_score_image, (v["coords"][1], v["coords"][0]),(v["coords"][3], v["coords"][2]),
                      col(aggregate_alignment_score_val),
                      thickness=cv2.FILLED)


    cv2.imwrite(op_folder+"area.png",area_image)
    cv2.imwrite(op_folder+"right_distance_image.png", right_distance_image)
    cv2.imwrite(op_folder+"left_distance_image.png", left_distance_image)
    cv2.imwrite(op_folder+"top_distance_image.png", top_distance_image)
    cv2.imwrite(op_folder+"bottom_distance_image.png", bottom_distance_image)

    cv2.imwrite(op_folder+"top_left_aligned_image.png", top_left_aligned_image)
    cv2.imwrite(op_folder+"continous_top_left_aligned_image.png", continous_top_left_aligned_image)
    cv2.imwrite(op_folder+"bottom_left_aligned_image.png", bottom_left_aligned_image)
    cv2.imwrite(op_folder+"continous_bottom_left_aligned_image.png", continous_bottom_left_aligned_image)
    cv2.imwrite(op_folder+"top_right_aligned_image.png", top_right_aligned_image)
    cv2.imwrite(op_folder+"continous_top_right_aligned_image.png", continous_top_right_aligned_image)
    cv2.imwrite(op_folder+"bottom_right_aligned_image.png", bottom_right_aligned_image)
    cv2.imwrite(op_folder+"continous_bottom_right_aligned_image.png", continous_bottom_right_aligned_image)
    cv2.imwrite(op_folder+"top_aligned_image.png", top_aligned_image)
    cv2.imwrite(op_folder+"top_continous_aligned_image.png", top_continous_aligned_image)
    cv2.imwrite(op_folder+"bottom_aligned_image.png", bottom_aligned_image)
    cv2.imwrite(op_folder+"bottom_continous_aligned_image.png", bottom_continous_aligned_image)

    cv2.imwrite(op_folder+"left_aligned_image.png", left_aligned_image)
    cv2.imwrite(op_folder+"left_continous_aligned_image.png", left_continous_aligned_image)
    cv2.imwrite(op_folder+"right_aligned_image.png", right_aligned_image)
    cv2.imwrite(op_folder+"right_continous_aligned_image.png", right_continous_aligned_image)
    cv2.imwrite(op_folder+"fully_aligned_image.png", fully_aligned_image)
    cv2.imwrite(op_folder+"fully_continous_aligned_image.png", fully_continous_aligned_image)

    cv2.imwrite(op_folder + "left_alignment_score_image.png", left_alignment_score_image)
    cv2.imwrite(op_folder + "right_alignment_score_image.png", right_alignment_score_image)
    cv2.imwrite(op_folder + "aggregate_alignment_score_image.png", aggregate_alignment_score_image)

def check_one_median_pass_through_rows(p1, _p2, center_area=1):
    h = _p2[2] - _p2[0]
    padding = h * (1 - center_area)
    p2 = (
        (_p2[1], _p2[0] + padding),
        (_p2[3], _p2[2] - padding),
        _p2[3]
    )
    return p1[0] <= (p2[0] + p2[2]) / 2 <= p1[2]  # or p2[0][1] <= (p1[1][1] + p1[0][1]) / 2 <= p2[1][1]


class MLFeatures():
    obj_list=[]
    def __init__(self,image,evidence):
        self.image=image
        self.evidence=evidence
        self.metrics={}

    @staticmethod
    def intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    @staticmethod
    def is_rect_overlapping(rect1, rect2):  # (x1, y1, x1b, y1b), (x2, y2, x2b, y2b)):
        left = rect2[2] < rect1[0]  # [x2b < x1
        right = rect1[2] < rect2[0]  # x1b < x2
        bottom = rect2[3] < rect1[1]  # y2b < y1
        top = rect1[3] < rect2[1]  # y1b < y2
        if top or left or bottom or right:
            return False
        else:  # rectangles intersect
            return True

    @staticmethod
    def get_whitespace_above(word_patch_coordinates, top_neighbour):
        distance = abs(word_patch_coordinates[0] - top_neighbour["assembled_result"][0][2])
        return distance

    @staticmethod
    def get_whitespace_below(word_patch_coordinates, bottom_neighbour):
        distance = abs(bottom_neighbour["assembled_result"][0][0] - word_patch_coordinates[2])
        return distance

    @staticmethod
    def get_whitespace_on_left(word_patch_coordinates, left_neighbour):
        distance = abs(word_patch_coordinates[1] - left_neighbour["assembled_result"][0][3])
        return distance

    @staticmethod
    def get_whitespace_on_right(word_patch_coordinates, right_neighbour):

        distance = abs(right_neighbour["assembled_result"][0][1] - word_patch_coordinates[3])
        return distance

    @staticmethod
    def get_normalised_score(value):
        A = 1
        b = 1
        return (A * math.exp(-1 * b * value))

    @staticmethod
    def degree_of_alignment(word_patch_coordinates,vertical_patches):
        left_score=0
        right_score=0
        aggregate_score=0
        for top_value in vertical_patches["top"]:
            #l-l
            left_value=abs(word_patch_coordinates[1]- top_value["assembled_result"][0][1])
            left_score= left_score + MLFeatures.get_normalised_score(left_value)

            right_value = abs(word_patch_coordinates[3] - top_value["assembled_result"][0][3])
            right_score = right_score + MLFeatures.get_normalised_score(right_value)

            aggregate_score=aggregate_score + MLFeatures.get_normalised_score(left_value+right_value)


        for bottom_value in vertical_patches["bottom"]:
            left_value=abs(word_patch_coordinates[1]- bottom_value["assembled_result"][0][1])
            left_score= left_score + MLFeatures.get_normalised_score(left_value)

            right_value = abs(word_patch_coordinates[3] - bottom_value["assembled_result"][0][3])
            right_score = right_score + MLFeatures.get_normalised_score(right_value)

            aggregate_score = aggregate_score + MLFeatures.get_normalised_score(left_value + right_value)

        return left_score, right_score , aggregate_score

    def get_vertical_patches(self,word_patch_coordinates):
        """
        Gather all word patches in vertical area of a word_patch coordinate
        :param word_patch_coordinates: coordinate of word patch (y1,x1,y2,x2)
        :return: dictionary :{top:[],bottom:[]}
        """
        data={"top":[],"bottom":[]}
        # top_box =[0,left,top+1,right]
        top_box=[0,word_patch_coordinates[1],word_patch_coordinates[0]+1,word_patch_coordinates[3]]
        bottom_box=[word_patch_coordinates[2]-1,word_patch_coordinates[1],self.image.shape[0],word_patch_coordinates[3]]
        for k, v in self.evidence["evidence_words"].items():

            _top = v["assembled_result"][0][0]
            _bottom = v["assembled_result"][0][2]
            word_patch_midpoint_y = (word_patch_coordinates[0] + word_patch_coordinates[2]) / 2
            if (_top + _bottom)/2 < word_patch_midpoint_y:
               if MLFeatures.is_rect_overlapping(top_box,v["assembled_result"][0]) and str(word_patch_coordinates)!= str(v["assembled_result"][0]):
                    data["top"].append(v)
            else:
                if MLFeatures.is_rect_overlapping(bottom_box,v["assembled_result"][0]) and str(word_patch_coordinates)!= str(v["assembled_result"][0]):
                    data["bottom"].append(v)

        data["bottom"]=sorted(data["bottom"],
                               key=lambda x: (x["assembled_result"][0][0] + x["assembled_result"][0][2]) / 2)
        data["top"] = sorted(data["top"],
                                key=lambda x: -(x["assembled_result"][0][0] + x["assembled_result"][0][2]) / 2)

        return data

    def get_horizontal_patches(self,word_patch_coordinates):
        """
        Gather all word patches in vertical area of a word_patch coordinate
        :param word_patch_coordinates: coordinate of word patch (y1,x1,y2,x2)
        :return: dictionary :{left:[],right:[]}
        """
        data={"left":[],"right":[]}
        left_box=[word_patch_coordinates[0],0,word_patch_coordinates[2],word_patch_coordinates[1]+1]

        right_box=[word_patch_coordinates[0],word_patch_coordinates[3]-1,word_patch_coordinates[2],self.image.shape[1]]

        for k, v in self.evidence["evidence_words"].items():
            if (v["assembled_result"][0][1]+v["assembled_result"][0][3])/2 <(word_patch_coordinates[1]+word_patch_coordinates[3])/2:

                if MLFeatures.is_rect_overlapping(left_box,v["assembled_result"][0]) and str(word_patch_coordinates)!= str(v["assembled_result"][0]):
                    data["left"].append(v)
            else:
                if MLFeatures.is_rect_overlapping(right_box,v["assembled_result"][0]) and str(word_patch_coordinates)!= str(v["assembled_result"][0]):
                    data["right"].append(v)

        data["right"] = sorted(data["right"],
                                key=lambda x: (x["assembled_result"][0][1] + x["assembled_result"][0][3]) / 2)
        data["left"] = sorted(data["left"],
                             key=lambda x: -(x["assembled_result"][0][1] + x["assembled_result"][0][3]) / 2)

        return data

    def get_left_aligned_words(self,word_patch_coordinates,list_of_patches):

        def is_in_left_range(patch_coordinate):
            pixel_threshold=5
            if patch_coordinate[1]<=(word_patch_coordinates[1]+(pixel_threshold/2)) and patch_coordinate[1] >= (word_patch_coordinates[1]-(pixel_threshold/2)):
                return True
            return False

        sorted_bottom=list_of_patches["bottom"]
        sorted_top=list_of_patches["top"]
        bottom_left_aligned=[]
        continous_bottom_left_aligned=[]
        counter=0
        for index,item in enumerate(sorted_bottom):
            if is_in_left_range(item["assembled_result"][0]):
                bottom_left_aligned.append(item)
                if index==counter:
                    continous_bottom_left_aligned.append(item)
                    counter=counter+1

        top_left_aligned = []
        continous_top_left_aligned = []
        counter = 0
        for index, item in enumerate(sorted_top):
            if is_in_left_range(item["assembled_result"][0]):
                top_left_aligned.append(item)
                if index == counter:
                    continous_top_left_aligned.append(item)
                    counter = counter + 1

        return top_left_aligned, continous_top_left_aligned, bottom_left_aligned, continous_bottom_left_aligned

    def get_right_aligned_words(self,word_patch_coordinates,list_of_patches):
        def is_in_right_range(patch_coordinate):
            pixel_threshold=10
            if patch_coordinate[3]<=(word_patch_coordinates[3]+(pixel_threshold/2)) and patch_coordinate[3] >= (word_patch_coordinates[3]-(pixel_threshold/2)):
                return True
            return False

        sorted_bottom=list_of_patches["bottom"]
        sorted_top=list_of_patches["top"]
        bottom_right_aligned=[]
        continous_bottom_right_aligned=[]
        counter=0
        for index,item in enumerate(sorted_bottom):
            if is_in_right_range(item["assembled_result"][0]):
                bottom_right_aligned.append(item)
                if index==counter:
                    continous_bottom_right_aligned.append(item)
                    counter=counter+1

        top_right_aligned = []
        continous_top_right_aligned = []
        counter = 0
        for index, item in enumerate(sorted_top):
            if is_in_right_range(item["assembled_result"][0]):
                top_right_aligned.append(item)
                if index == counter:
                    continous_top_right_aligned.append(item)
                    counter = counter + 1

        return top_right_aligned, continous_top_right_aligned, bottom_right_aligned, continous_bottom_right_aligned

    def get_left_and_right_aligned_words(self,top_left_aligned,  continous_top_left_aligned,  bottom_left_aligned,  continous_bottom_left_aligned,
                                              top_right_aligned, continous_top_right_aligned, bottom_right_aligned, continous_bottom_right_aligned):
        top_aligned =    MLFeatures.intersection(top_left_aligned,top_right_aligned)
        # set(top_left_aligned) & set(top_right_aligned)
        bottom_aligned = MLFeatures.intersection(bottom_left_aligned,bottom_right_aligned)
        # set(bottom_left_aligned) & set(bottom_right_aligned)
        counter=0
        top_continous_aligned=[]
        for index,(l,r) in enumerate(zip(continous_top_left_aligned,continous_top_right_aligned)):
            if l==r and index==counter:
                counter=counter+1
                top_continous_aligned.append(l)

        counter=0
        bottom_continous_aligned=[]
        for index,(l,r) in enumerate(zip(continous_bottom_left_aligned,continous_bottom_right_aligned)):
            if l==r and index==counter:
                counter=counter+1
                bottom_continous_aligned.append(l)
        return top_aligned,top_continous_aligned,bottom_aligned,bottom_continous_aligned

    def get_white_space_around_patch(self,word_patch_coordinates,top_neighbour,bottom_neighbour,left_neighbour,right_neighbour):
        top=MLFeatures.get_whitespace_above(word_patch_coordinates, top_neighbour)

        bottom=MLFeatures.get_whitespace_below(word_patch_coordinates, bottom_neighbour)
        left=MLFeatures.get_whitespace_on_left(word_patch_coordinates, left_neighbour)
        right=MLFeatures.get_whitespace_on_right(word_patch_coordinates, right_neighbour)
        word_height = word_patch_coordinates[2] - word_patch_coordinates[0]
        #TODO why do we need absolute below, review this code with Kevin
        #word_width = word_patch_coordinates[3] - word_patch_coordinates[1]
        #word_and_padding_area = (top + bottom + abs(word_height)) * (left + right + abs(word_width))
        #white_space = word_and_padding_area - abs((word_height) * (word_width))

        white_space = left+right+bottom+top


        norm_dict={"top":top/self.image.shape[0],
                   "bottom":bottom/self.image.shape[0],
                   "left":left/self.image.shape[1],
                   "right":right/self.image.shape[1],
                   "area":white_space/(self.image.shape[0]*self.image.shape[1])}

        return norm_dict

    def relative_score(self):
        pass
    @staticmethod
    #Params : data array of data
    #Params : m Number of standard deviations above (and below) which data is excluded
    def get_ranges_for_outlier(data,m=2):
        elements = np.array(data)
        mean = np.mean(elements, axis=0)
        sd = np.std(elements, axis=0)
        final_list = [x for x in data if (x > mean - m * sd)]
        final_list = [x for x in final_list if (x < mean + m * sd)]

        if len(final_list) == 0:
            return 0,0

        return min(final_list), max(final_list)

    def run(self):
        print("Creating features for each word")
        data_dict={}
        top_distance_array=[]
        bottom_distance_array=[]
        left_distance_array=[]
        right_distance_array=[]
        area_distance_array=[]
        for k, v in self.evidence["evidence_words"].items():
            word = v["assembled_result"]
            vertical_data=self.get_vertical_patches(word[0])

            left_alignment_score, right_alignment_score, aggregate_alignment_score=MLFeatures.degree_of_alignment(word_patch_coordinates=
                                                                                                                  word[0], vertical_patches=vertical_data)
            top=vertical_data["top"][0] if len(vertical_data["top"]) !=0 else {"assembled_result":[[0, word[0][1], 0,
                                                                                                    word[0][3]]]}
            bottom=vertical_data["bottom"][0]if len(vertical_data["bottom"]) !=0 else {"assembled_result":[[self.image.shape[0],
                                                                                                            word[0][1], self.image.shape[0],
                                                                                                            word[0][3]]]}
            horizontal_data=self.get_horizontal_patches(word[0])
            left=horizontal_data["left"][0] if len(horizontal_data["left"]) !=0 else {"assembled_result":[[word[0][0], 0,
                                                                                                           word[0][2], 0]]}
            right=horizontal_data["right"][0] if len(horizontal_data["right"]) !=0 else {"assembled_result":[[
                                                                                                                 word[0][0], self.image.shape[1],
                                                                                                                 word[0][2], self.image.shape[1]]]}
            distance_array=self.get_white_space_around_patch(word[0], top, bottom, left, right)
            top_distance_array.append(distance_array["top"])
            bottom_distance_array.append(distance_array["bottom"])
            left_distance_array.append(distance_array["left"])
            right_distance_array.append(distance_array["right"])
            area_distance_array.append(distance_array["area"])
            # data_dict[str(v["assembled_result"][0])]={"dist":distance_array,"coords":v["assembled_result"][0]}
            top_left_aligned,  continous_top_left_aligned,  bottom_left_aligned,  continous_bottom_left_aligned  = self.get_left_aligned_words(list_of_patches=vertical_data, word_patch_coordinates=
            word[0])
            top_right_aligned, continous_top_right_aligned, bottom_right_aligned, continous_bottom_right_aligned = self.get_right_aligned_words(list_of_patches=vertical_data, word_patch_coordinates=
            word[0])
            top_aligned, top_continous_aligned, bottom_aligned, bottom_continous_aligned=self.get_left_and_right_aligned_words( top_left_aligned,  continous_top_left_aligned,  bottom_left_aligned,  continous_bottom_left_aligned,top_right_aligned, continous_top_right_aligned, bottom_right_aligned, continous_bottom_right_aligned)

            page_start=0

            page_end=0

            if len(horizontal_data["left"]) == 0:
                page_start = 1

            if len(horizontal_data["right"]) == 0:
                page_end = 1


            neighbours=[vertical_data["top"][0]["assembled_result"][0] if len(vertical_data["top"])>0 else [] ,
                        vertical_data["bottom"][0]["assembled_result"][0] if len(vertical_data["bottom"])>0 else [],
                        horizontal_data["left"][0]["assembled_result"][0] if len(horizontal_data["left"])>0 else [],
                        horizontal_data["right"][0]["assembled_result"][0] if len(horizontal_data["right"])>0 else []]



            data_dict[str(word[0])]={"top_left_aligned":len(top_left_aligned),
                                                      "continous_top_left_aligned" :len(continous_top_left_aligned),
                                                      "bottom_left_aligned":len(bottom_left_aligned),
                                                      "continous_bottom_left_aligned":len(continous_bottom_left_aligned),
                                                      "top_right_aligned":len(top_right_aligned),
                                        "continous_top_right_aligned":len(continous_top_right_aligned),
                                        "bottom_right_aligned":len(bottom_right_aligned),
                                        "continous_bottom_right_aligned":len(continous_bottom_right_aligned),
                                                      "top_aligned":len(top_aligned),
                                                      "top_continous_aligned":len(top_continous_aligned),
                                                      "bottom_aligned":len(bottom_aligned),
                                                      "bottom_continous_aligned":len(bottom_continous_aligned),
                                                      "dist": distance_array,
                                                      "coords": word[0],
                                                      "left_aligned":len(top_left_aligned)+len(bottom_left_aligned),
                                                      "left_continous_aligned":len(continous_top_left_aligned)+len(continous_bottom_left_aligned),
                                                      "right_aligned":len(top_right_aligned)+len(bottom_right_aligned),
                                                      "right_continous_aligned":len(continous_top_right_aligned)+len(continous_bottom_right_aligned),
                                        "fully_aligned":len(top_aligned)+len(bottom_aligned),
                                        "fully_continous_aligned":len(top_continous_aligned)+len(bottom_continous_aligned),
                                        "left_alignment_score":left_alignment_score,
                                        "right_alignment_score":right_alignment_score,
                                        "aggregate_alignment_score":aggregate_alignment_score,
                                        "neighbours":neighbours,
                                        "page_start":page_start,
                                        "page_end":page_end
                                        }

        top_min,top_max = MLFeatures.get_ranges_for_outlier(top_distance_array)
        bottom_min, bottom_max = MLFeatures.get_ranges_for_outlier(bottom_distance_array)
        right_min, right_max = MLFeatures.get_ranges_for_outlier(right_distance_array)
        left_min, left_max = MLFeatures.get_ranges_for_outlier(left_distance_array)
        area_min, area_max = MLFeatures.get_ranges_for_outlier(area_distance_array)


        for k,v in data_dict.items():
            temp={}
            if v["dist"]["top"] >= top_min and v["dist"]["top"] <= top_max:
                temp["top"]=(v["dist"]["top"] - top_min)/(top_max - top_min)
                # temp["top"] =MLFeatures.get_normalised_score((v["dist"]["top"] - top_min) / (top_max - top_min))
            elif v["dist"]["top"] < top_min:
                # temp["top"] = (top_min - top_min) / (top_max - top_min)
                temp["top"]=0
                # temp["top"] = MLFeatures.get_normalised_score(0)
            elif v["dist"]["top"] > top_max:
                # temp["top"] = (top_max - top_min) / (top_max - top_min)
                temp["top"]=1
                # temp["top"] = MLFeatures.get_normalised_score(1)

            if v["dist"]["bottom"] >= bottom_min and v["dist"]["bottom"] <= bottom_max:
                temp["bottom"]=(v["dist"]["bottom"] - bottom_min)/(bottom_max - bottom_min)
                # temp["bottom"] = MLFeatures.get_normalised_score((v["dist"]["bottom"] - bottom_min) / (bottom_max - bottom_min))
            elif v["dist"]["bottom"] < bottom_min:
                temp["bottom"]=0
                # temp["bottom"] = MLFeatures.get_normalised_score(0)
            elif v["dist"]["bottom"] > bottom_max:
                temp["bottom"]=1
                # temp["bottom"] = MLFeatures.get_normalised_score(1)

            if v["dist"]["right"] >= right_min and v["dist"]["right"] <= right_max:
                temp["right"]=(v["dist"]["right"] - right_min)/(right_max - right_min)
                # temp["right"] = MLFeatures.get_normalised_score((v["dist"]["right"] - right_min) / (right_max - right_min))
            elif v["dist"]["right"] < right_min:
                temp["right"]=0
                # temp["right"] = MLFeatures.get_normalised_score(0)
            elif v["dist"]["right"] > right_max:
                temp["right"]=1
                # temp["right"] = MLFeatures.get_normalised_score(1)

            if v["dist"]["left"] >= left_min and v["dist"]["left"] <= left_max:
                temp["left"]=(v["dist"]["left"] - left_min)/(left_max - left_min)
                # temp["left"] = MLFeatures.get_normalised_score((v["dist"]["left"] - left_min) / (left_max - left_min))
            elif v["dist"]["left"] < left_min:
                temp["left"]=0
                # temp["left"] = MLFeatures.get_normalised_score(0)
            elif v["dist"]["left"] > left_max:
                temp["left"]=1
                # temp["left"] = MLFeatures.get_normalised_score(1)

            if v["dist"]["area"] >= area_min and v["dist"]["area"] <= area_max:
                temp["area"]=(v["dist"]["area"] - area_min)/(area_max - area_min)
                # temp["area"] = MLFeatures.get_normalised_score((v["dist"]["area"] - area_min) / (area_max - area_min))
            elif v["dist"]["area"] < area_min:
                temp["area"]=0
                # temp["area"] = MLFeatures.get_normalised_score(0)
            elif v["dist"]["area"] > area_max:
                temp["area"]=1
                # temp["area"] = MLFeatures.get_normalised_score(1)

            v["dist"]=temp
            # print(data_dict[k]["dist"])
            # print("final_feature_done")
        print("Completed creating features for each word")

        return data_dict

# from  ocr_entities.words_list import WordsList


def get_evidence(evidence_df, key):
    evidence_for_key = evidence_df[evidence_df[Constants.IMAGE_KEY] == key]
    cordinates  = evidence_for_key["Coordinates"].values
    labels  = evidence_for_key["Text"].values
    word_id = 0
    words_dict =dict()
    for c,l in zip(cordinates,labels):
        word_key = "word_" + str(word_id)
        words_dict[word_key] = {"assembled_result":[eval(c),l]}
        word_id += 1

    return {"evidence_words":words_dict}


        # "evidence_words": {"word_0": {"assembled_result": [[189, 198, 201, 823], "", 0.3, "IN_LINE"],
        #                               "tesseract": [[189, 198, 201, 823], ""], "rzt_ocr": [], "google_cloud_vision": [],
        #                               "true_pdf": 0},
        #                    "word_1": {"assembled_result": [[185, 242, 188, 267], "03", 0.3, "IN_LINE"], "tesseract": [],
        #                               "rzt_ocr": [], "google_cloud_vision": [[185, 242, 188, 267], "03"],
        #                               "true_pdf": 0},

def get_page_coordinates_from_coordinates_data(coordinates_data_dict):
    """
    converts x,y,width,height to top,left,bottom,right
    :param coordinates_data_dict: dictionary having x,y,width,height values
    :return: tuple in top,left,bottom,right format
    """
    if type(coordinates_data_dict)==list:
        return coordinates_data_dict
    return (coordinates_data_dict['y'], coordinates_data_dict['x'],
            (coordinates_data_dict['y'] + coordinates_data_dict['height']),
            (coordinates_data_dict['x'] + coordinates_data_dict['width']))


def get_coordinates_data_from_page_coordinates(page_coordinates):
    """
    converts top,left,bottom,right to x,y,width,height
    :param page_coordinates: list in top,left,bottom,right format
    :return: dictionary as x,y,width,height
    """
    # print("page_coordinates",page_coordinates)
    return {'x': min(page_coordinates[1],page_coordinates[3]), 'y': min(page_coordinates[0],page_coordinates[2]), 'width': abs(page_coordinates[3] - page_coordinates[1]),
            'height': abs(page_coordinates[2] - page_coordinates[0])}


def get_test_evidence(json_path):
    with open(json_path, "r")as gt_file:
        gt = json.load(gt_file)
    word_id = 0
    words_dict = dict()
    coordinate_to_text_map={}
    for each_gt in gt['words']:
        c=get_page_coordinates_from_coordinates_data(each_gt['coordinates'])
        l=each_gt['label']
        coordinate_to_text_map [c]=l
        word_key = "word_" + str(word_id)
        words_dict[word_key] = {"assembled_result": [c, l]}
        word_id += 1

    return {"evidence_words": words_dict},coordinate_to_text_map


if __name__ == "__main__":
    #
    # #Input folder paths
    # dataset_name = "invoice"
    # IMAGE_PATH =IMAGE_PATH
    #
    # # with open(DATASET_PATH  + "/ground_truth.json", "r")as gt_file:
    # #     gt = json.load(gt_file)
    #
    #
    # # evidence_df = pd.read_csv(DATASET_PATH+"/Features.tsv",sep="\t")
    #
    # #Output folder paths
    # op_folder = DATASET_PATH + "/word_features/"
    # op_folder='/Users/amandubey/Documents/RZT/Tables_test_set/output_csv_folder/'
    # #op_image_folder = op_folder + "images/"
    #
    # #Open output file and write the column titles



    # Iterate over each image
    # keys = [k.rsplit(".",1)[0] for k in gt.keys()]
    # WORD_IMAGE_PATH= '/Users/amandubey/Documents/RZT/Tables_test_set/input_folder/words_image/'
    JSON_FOLDER= '/Users/amandubey/Documents/RZT/Tables_test_set/input_folder/evidence_folder/'
    # a=2
    json_files=os.listdir(JSON_FOLDER)
    # for key in keys:
    for json_path in json_files:
        if json_path.startswith('.'):
            continue
        key=(json_path.split('.json'))[0]

        json_path= JSON_FOLDER + key + '.json'

        _image_path = IMAGE_PATH + key+".jpg"
        image = cv2.imread(_image_path)
        evidence,coordinate_to_text_map = get_test_evidence(json_path=json_path)
        # coordinate_to_text_map = { str(word[0]):word[1] for word in words_list.words() }

        Image_features = MLFeatures(image=image,evidence=evidence)

        d=Image_features.run()

        #Get the gt - skip during testing
        # table_cordinates = []
        # for table_cord in gt[key+".jpg"]["table"]:
        #     table_cordinates.append([table_cord[1], table_cord[0], table_cord[3],table_cord[2]])
        # for cord in table_cordinates:
        #     cv2.rectangle(image,(cord[1],cord[0]),(cord[3],cord[2]),(0,255,0))

        '[t,l,b,r]'

        # Open output file and write the column titles
        f = open(WORD_FEATURES_PATH + key+"_Features.tsv", "w", encoding="utf-8")
        writer = csv.writer(f, delimiter="\t")
        columns = ['Filename','Text','Coordinates','Number_of_words_left_aligned_above','Number_of_words_left_aligned_above_contigous',
                   'Number_of_words_left_aligned_below_contigous','Number_of_words_left_aligned_below',
                   'Number_of_words_right_aligned_above','Number_of_words_right_aligned_above_contigous',
                   'Number_of_words_right_aligned_below','Number_of_words_right_aligned_below_contigous',
                   'Number_of_words_left_and_right_aligned_above','Number_of_words_left_and_right_aligned_above_contigous',
                   'Number_of_words_left_and_right_aligned_below','Number_of_words_left_and_right_aligned_below_contigous',
                   'Total_number_of_left_aligned_words','Total_number_of_left_aligned_words_contigous',
                   'Total_number_of_right_aligned_words','Total_number_of_right_aligned_words_contigous',
                   'Number_of_words_left_and_right_aligned_above_and_below',
                   'Number_of_words_left_and_right_aligned_above_and_below_contigous',
                   'Left_alignment_score','Right_alignment_score',
                   'Total_score_for _all_words_above_and_below',
                   'Total_whitespace_around_the_word',
                   'Whitespace_distance_above_normalized',
                   'Whitespace_distance_below_normalized',
                   'Whitespace_distance_to_the_left_of_normalized',
                   'Whitespace_distance_to_the_right_of_normalized'
                   # "Label"
                   ]
        writer.writerow(columns)

        # for each word in page
        for k, v in d.items():
            label = 0
            _k = eval(k)
            cv2.rectangle(image, (_k[1], _k[0]), (_k[3], _k[2]), (0, 0, 255), 2)

            # for table_id,table in enumerate(table_cordinates):
            #     if MLFeatures.is_rect_overlapping(v["coords"], table):
            #         label = table_id + 1
            #         break
            values =[key,
                coordinate_to_text_map[eval(k)],
                k,
                v["top_left_aligned"],
                v["continous_top_left_aligned"],
                v["bottom_left_aligned"],
                v["continous_bottom_left_aligned"],
                v["top_right_aligned"],
                v["continous_top_right_aligned"],
                v["bottom_right_aligned"],
                v["continous_bottom_right_aligned"],
                v["top_aligned"],
                v["top_continous_aligned"],
                v["bottom_aligned"],
                v["bottom_continous_aligned"],
                v["left_aligned"],
                v["left_continous_aligned"],
                v["right_aligned"],
                v["right_continous_aligned"],
                v["fully_aligned"],
                v["fully_continous_aligned"],
                v["left_alignment_score"],
                v["right_alignment_score"],
                v["aggregate_alignment_score"],
                v["dist"]["area"],
                v["dist"]["top"],
                v["dist"]["bottom"],
                v["dist"]["left"],
                v["dist"]["right"] ,
                label ]
            writer.writerow(values)
        #cv2.imwrite(op_image_folder + key+".jpg", image)
        f.close()
        cv2.imwrite(WORD_IMAGE_PATH + key + '.jpg', image)
        # print()
    # print()


    

