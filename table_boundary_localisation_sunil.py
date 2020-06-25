from ocr_pattern_hypothesis.utils import frame_utils
from ocr_pattern_hypothesis.frames.basic_frames import Word

# STRUCTURE FRAMES
from ocr_pattern_hypothesis.frames.structure.engine import StructureEngine
from ocr_pattern_hypothesis.frames.structure.text import  TextLine
import cv2
import json
import numpy as np
# import cv2
import json
from collections import Counter
from collections import OrderedDict
from dijkstar import Graph, find_path
import math
from sklearn.cluster import MeanShift
import random

from config.paths import PREDICTION_RESULTS_PATH, IMAGE_PATH, JSON_FOLDER,WORD_FEATURES_PATH,\
    PREDICTION_IMAGES_RESULT_PATH,POSTPROCESSING_RESULT_PATH,PREDICTION_JSON_RESULT_PATH

colors =[(0, 0, 255), (0, 255, 0),(0,0,0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]


class TableLocalisation():

    def __init__(self,image,evidence,table_evidence,image_name):
        self.image_name = image_name
        self.image=image
        self.evidence=evidence
        self.table_evidence=table_evidence

    @staticmethod
    def is_rect_overlapping(rect1, rect2):  #[]
        left = rect2[2] < rect1[0]  # [x2b < x1
        right = rect1[2] < rect2[0]  # x1b < x2
        bottom = rect2[3] < rect1[1]  # y2b < y1
        top = rect1[3] < rect2[1]  # y1b < y2
        if top or left or bottom or right:
            return False
        else:  # rectangles intersect
            return True




    def _get_textlines1(self):

        s_engine = StructureEngine((
            TextLine.generate,
        ))
        word_patches_dict = {}
        structures = []

        for each_evidence in self.evidence['words']:
            # label = word_dict['assembled_result'][1]
            # coordinates = (word_dict['assembled_result'][0][0], word_dict['assembled_result'][0][1],
            #                word_dict['assembled_result'][0][2], word_dict['assembled_result'][0][3])
            for each_evidence in evidence['words']:
                label = str(each_evidence['label'])
                coordinates = (each_evidence['coordinates']['y'], each_evidence['coordinates']['x'],
                               (each_evidence['coordinates']['height'] + each_evidence['coordinates']['y']),
                               (each_evidence['coordinates']['width'] + each_evidence['coordinates']['x']))
                label_word = label
                word_patches_dict[coordinates] = label_word
        # print(word_patches_dict)
        try:
            structures = s_engine.run(self.image, word_args=(word_patches_dict,))

        except IndexError:
            structures = []
        structures = structures.filter(TextLine)
        return structures

    def _get_textlines(self):
        s_engine = StructureEngine((
            TextLine.generate,
        ))
        word_patches_dict = {}
        for k, v in self.evidence['evidence_words'].items():
            c = v["assembled_result"][0]
            label = v["assembled_result"][1]

            coordinates = (
                c[0], c[1],
                c[2], c[3]
            )
            word_patches_dict[coordinates] = label

        try:
            structures = s_engine.run(self.image, word_args=(word_patches_dict,))
        except IndexError:
            structures = []
        structures=structures.filter(TextLine)
        return structures


    def _get_connected_neighbour(self,feed_patches,blockers):

        def fetch_graph(space_dict, ver_gap=3300, hor_gap=2550):

            # f, axarr = plt.subplots(2, 2)

            xu = sorted(space_dict["top"])
            xb = sorted(space_dict["bottom"])
            xv = set(xu) | set(xb)
            xv = sorted(list(xv))
            freqv = []
            for item in xv:
                sum = 0
                if item in space_dict["top"]:
                    sum = sum + space_dict["top"][item]
                if item in space_dict["bottom"]:
                    sum = sum + space_dict["bottom"][item]
                freqv.append(sum)
            # yv = [i + 1 for i in range(len(xv))]
            # axarr[0, 0].bar(yv, freqv)
            # axarr[0, 0].set_title('Vertical freq')
            # axarr[0, 1].bar(yv, xv)
            # axarr[0, 1].set_title('Vertical space')

            xl = sorted(space_dict["left"])
            xr = sorted(space_dict["right"])
            xh = set(xl) | set(xr)
            xh = sorted(list(xh))
            freqh = []
            for item in xh:
                sum = 0
                if item in space_dict["left"]:
                    sum = sum + space_dict["left"][item]
                if item in space_dict["right"]:
                    sum = sum + space_dict["right"][item]
                freqh.append(sum)

            # yh = [i + 1 for i in range(len(xh))]
            # axarr[1, 0].bar(yh, freqh)
            # axarr[1, 0].set_title('Horizontal freq')
            # axarr[1, 1].bar(yh, xh)
            # axarr[1, 1].set_title('Horizontal space')

            # f.subplots_adjust(hspace=0.3)
            # plt.show()
            temp = 0
            hor_thresh = 0
            ver_thresh = 0

            for enum, val in enumerate(freqh):
                if xh[enum] < int(hor_gap / 4):
                    if temp > val:
                        hor_thresh = xh[enum]
                        break
                    else:
                        temp = val
            temp = 0
            for enum, val in enumerate(freqv):
                if xv[enum] < int(ver_gap / 6):
                    if temp > val:
                        ver_thresh = xv[enum]
                        break
                    else:
                        temp = val

            return hor_thresh, ver_thresh

        block_img = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
        for word_patch in feed_patches:
            block_img[word_patch[0]:word_patch[2],
            word_patch[1]:word_patch[3]] = 255

        for blk in blockers:
            block_img[blk[0]:blk[2],
            blk[1]:blk[3]] = 128

        upper_thresh = 10000
        lower_thresh = 10000
        left_thresh = 10000
        right_thresh = 10000

        new_block_img = np.copy(block_img)
        # tot_len = len(evidences["evidence_words"])
        itter = 1
        space_values = {"top": [], "bottom": [], "left": [], "right": []}

        for word_patch in feed_patches:
        # for k, v in evidences["evidence_words"].items():
            # print(itter,"out of ",tot_len)
            itter = itter + 1
            upper = word_patch[0]
            lower = word_patch[2]
            left = word_patch[1]
            right = word_patch[3]
            if (upper > lower):
                temp = upper
                upper = lower
                lower = temp

            if (left > right):
                temp = left
                left = right
                right = temp

            flag = False
            upper_flag = True
            lower_flag = True
            x1 = upper - 1
            x2 = lower + 1

            while not flag:  # upper_flag and not lower_flag:
                if upper_flag and not flag:
                    if 255 in block_img[x1, left:(right + 1)]:
                        flag = True
                        lower_flag = False
                    elif x1 < 1 or 128 in block_img[x1, left:(right + 1)]:

                        upper_flag = False
                    else:
                        x1 = x1 - 1

                if lower_flag and not flag:
                    if 255 in block_img[x2, left:(right + 1)]:
                        flag = True
                        upper_flag = False
                    elif x2 > (block_img.shape[0] - 2) or 128 in block_img[x2, left:(right + 1)]:
                        lower_flag = False
                    else:
                        x2 = x2 + 1

                if not lower_flag and not upper_flag:
                    flag = True

            if upper_flag and flag:
                space_values["top"].append(upper - x1)

                if upper_thresh > upper - x1:
                    for coor in range(left, (right + 1)):
                        if block_img[x1][coor] == 255:
                            new_block_img[x1:upper, coor] = 100

            elif lower_flag and flag:
                space_values["bottom"].append(x2 - lower)
                if lower_thresh > x2 - lower:
                    for coor in range(left, (right)):
                        if block_img[x2][coor] == 255:
                            new_block_img[lower:x2, coor] = 100

            flag = False
            left_flag = True
            right_flag = True
            y1 = left - 1
            y2 = right + 1

            while not flag:  # upper_flag and not lower_flag:
                if left_flag and not flag:
                    if 255 in block_img[upper:(lower + 1), y1]:
                        flag = True
                        right_flag = False
                    elif y1 < 1 or 128 in block_img[upper:(lower + 1), y1]:
                        left_flag = False
                    else:
                        y1 = y1 - 1

                if right_flag and not flag:
                    if 255 in block_img[upper:(lower + 1), y2]:
                        flag = True
                        left_flag = False
                    elif y2 > (block_img.shape[1] - 2) or 128 in block_img[upper:(lower + 1), y2]:
                        right_flag = False
                    else:
                        y2 = y2 + 1
                if not left_flag and not right_flag:
                    flag = True

            if left_flag and flag:
                space_values["left"].append(left - y1)
                if left_thresh > left - y1:
                    for coor in range(upper, (lower + 1)):
                        if block_img[coor][y1] == 255:
                            new_block_img[coor, y1:left] = 100
            elif right_flag and flag:
                space_values["right"].append(y2 - right)
                if right_thresh > y2 - right:
                    for coor in range(upper, (lower + 1)):
                        if block_img[coor][y2] == 255:
                            new_block_img[coor, right:y2] = 100

        for blk in blockers:
            new_block_img[blk[0]:blk[2],
            blk[1]:blk[3]] = 0
        # cv2.imwrite("/home/amandubey/DATA_PART/Intermediate.jpg", new_block_img)
        cv2.imshow("Connected Image", new_block_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        space_dict = {}
        space_dict["top"] = OrderedDict(sorted(Counter(space_values["top"]).items(), key=lambda t: t[0]))
        space_dict["bottom"] = OrderedDict(sorted(Counter(space_values["bottom"]).items(), key=lambda t: t[0]))
        space_dict["left"] = OrderedDict(sorted(Counter(space_values["left"]).items(), key=lambda t: t[0]))
        space_dict["right"] = OrderedDict(sorted(Counter(space_values["right"]).items(), key=lambda t: t[0]))

        fetch_graph(space_dict,)

    def _create_patches_and_blockers(self,textlines):
        patches = []
        blockers = []
        for textline in textlines:
            word_cord_list=[str([word.coordinates[0][1],word.coordinates[0][0],word.coordinates[1][1],word.coordinates[1][0]]) for word in textline.contains["words"] ]
            f=0
            for wrd_coord in word_cord_list:
                try:
                    if self.table_evidence[wrd_coord] == 1:
                        f=1
                        break
                except KeyError:
                    print(wrd_coord)
                    continue
            if f == 0:
                blockers.append([textline.coordinates[0][1],textline.coordinates[0][0],textline.coordinates[1][1],textline.coordinates[1][0]])


        for word_coord,label in self.table_evidence.items():
            if label == '1':
                # word_coord = eval(word_coord)
                patches.append([int(c) for c in word_coord.replace(" ","")[1:-1].split(",")])

        return patches,blockers

    @staticmethod
    def get_absolute_distance_between_bloks(rect1, rect2):  # (x1, y1, x1b, y1b), (x2, y2, x2b, y2b)):
        left = rect2[3] - rect1[1]  # always -ve
        right = rect2[1] - rect1[3]  # always +ve
        bottom = rect2[0] - rect1[2]  # always +ve
        top = rect2[2] - rect1[0]  # always -ve
        if top < 0:
            if left < 0:
                return math.sqrt(math.pow(top, 2) + math.pow(left, 2))
            elif right > 0:
                return math.sqrt(math.pow(top, 2) + math.pow(right, 2))
            else:
                return abs(top)
        elif bottom > 0:
            if left < 0:
                return math.sqrt(math.pow(bottom, 2) + math.pow(left, 2))
            elif right > 0:
                return math.sqrt(math.pow(bottom, 2) + math.pow(right, 2))
            else:
                return abs(bottom)
        elif left < 0:
            return abs(left)
        elif right > 0:
            return abs(right)
        else:  # rectangles intersect
            return 0

    @staticmethod
    def centroid_distance(rect1, rect2):  # (x1, y1, x1b, y1b), (x2, y2, x2b, y2b)):
        # print(rect1)
        # print(rect2)
        # left = rect2[1][0] < rect1[0][0]  # [x2b < x1
        # right = rect1[1][0] < rect2[0][0]  # x1b < x2
        # bottom = rect2[1][1] < rect1[0][1]  # y2b < y1
        # top = rect1[1][1] < rect2[0][1]  # y1b < y2

        centroid1_y = (rect1[2] + rect1[0]) / 2
        # print(centroid1_y)
        centroid1_x = (rect1[3] + rect1[1]) / 2
        # print(centroid1_x)

        centroid2_y = (rect2[2] + rect2[0]) / 2
        # print(centroid2_y)
        centroid2_x = (rect2[3] + rect2[1]) / 2
        # print(centroid2_x)
        # print(math.sqrt(math.pow((centroid2_x-centroid1_x),2) + math.pow((centroid2_y-centroid1_y),2)))
        return math.sqrt(math.pow((centroid2_x-centroid1_x),2) + math.pow((centroid2_y-centroid1_y),2))




        # return math.sqrt(math.pow(centroid2[0] - centroid1[0], 2) +  math.pow(centroid2[1] - centroid1[1], 2))

    # @staticmethod
    def create_graphs(self,patches):
        g=Graph()
        for p1 in patches:
            for p2 in patches:
                if p1 == p2 :
                    continue
                g.add_edge(str(p1),str(p2),{'cost': TableLocalisation.get_absolute_distance_between_bloks(p1,p2)})
            # break
        cost_func = lambda u, v, e, prev_e: e['cost']
        data=[]
        print("creat graph computed cost func")
        print("Number of patches",len(patches))
        for i,p1 in enumerate(patches):
            points = frame_utils.calculate_all_points(((int(p1[1]), int(p1[0])),
                                                       (int(p1[3]), int(p1[2])), 0))
            frame_utils.draw_polygon(self.image, points,(0,0,0), thickness=4)
            local_temp = []
            for p2 in patches:
                if p1 == p2 :
                    local_temp.append(0)
                else:
                    local_temp.append(round(list(find_path(g, str(p1),str(p2), cost_func=cost_func))[-1],3))
                    # local_temp.append([round(TableLocalisation.centroid_distance(p1,p2),3)])
            if i % 50 == 0:
                print("Completed patch",i)
            data.append(local_temp)
        return np.asarray(data)

    def _get_coordinates_cc(self):
        pass

    def run(self):
        tl = (self._get_textlines1())
        patches,blockers=self._create_patches_and_blockers(textlines=tl)
        print('patches',len(patches),'blockers',len(blockers))
        print("Creating graph")
        data=(self.create_graphs(patches))
        print("Created graph")
        if len(data)==0 :
            return
        clustering = MeanShift().fit(data)

        colr_dict={}
        for i, class_label in enumerate(set(clustering.labels_)):
            colr_dict[class_label] = colors[i]

        # self._get_connected_neighbour(feed_patches=patches,blockers=blockers)
        for index,patch in enumerate(patches):
            points = frame_utils.calculate_all_points(((int(patch[1]), int(patch[0])),
                                           (int(patch[3]), int(patch[2])), 0))
            frame_utils.draw_polygon(self.image, points, colr_dict[list(clustering.labels_)[index]], thickness=4)

            # cv2.putText(self.image, str(data[index]), (int(patch[1]), int(patch[0])),
            #                 cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 0, 0))
        #
        # for blocker in blockers:
        #     points = frame_utils.calculate_all_points(((int(blocker[1]), int(blocker[0])),
        #                                    (int(blocker[3]), int(blocker[2])), 0))
        #     frame_utils.draw_polygon(self.image, points, (0, 255, 255), thickness=4)


import pandas as pd
import os
model_name = "crf_rf_4"
LABEL_COLUMN = "Is_Table_Word"

# image_for_prediction = '10.1.1.1.2006_3'
# image_for_prediction="10.1.1.160.506_4"
images_for_prediction = ["10.1.1.160.529_30"]
#images_for_prediction = ['10.1.1.1.2006_3',"10.1.1.160.506_4",]

if __name__ =="__main__":
    files = os.listdir(PREDICTION_IMAGES_RESULT_PATH)
    images_for_prediction = [f.rsplit(".",1)[0] for f in files]

    for image_for_prediction in images_for_prediction:
        try:
            if image_for_prediction == '5e98958f0570c934f8b2a005_0' or image_for_prediction.startswith('.') or image_for_prediction=='':
                print('skipping ', image_for_prediction)
                continue
            print('image_for_prediction',image_for_prediction)
            preds = {}
            # data=pd.read_csv(PREDICTION_RESULTS_PATH + model_name +"/page_level_predictions/"+image_for_prediction+".tsv",sep="\t")
            # for coord,label in zip(data["Coordinates"].tolist(),data[LABEL_COLUMN].tolist()):
            #     preds[coord]=label
            #
            jpg_ = IMAGE_PATH + "/" + image_for_prediction + ".jpg"
            print(jpg_)
            print(JSON_FOLDER + image_for_prediction + ".json")
            image = cv2.imread(jpg_)
            evidence_filename = JSON_FOLDER + image_for_prediction + ".json"
            preds = json.load(open(PREDICTION_JSON_RESULT_PATH + image_for_prediction + ".json"))
            evidence = json.load(open(evidence_filename))
            localisation = TableLocalisation(image, evidence, preds, image_for_prediction)
            print("Processing", image_for_prediction)
            localisation.run()
            cv2.imwrite(POSTPROCESSING_RESULT_PATH + image_for_prediction + ".jpg",
                        localisation.image)
        except Exception as e:
            raise e
            print('Error !!!!')

