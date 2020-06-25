import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score,accuracy_score
from common.todo_fix_this_later import Constants
from config.paths import PREDICTION_JSON_RESULT_PATH,PREDICTION_IMAGES_RESULT_PATH
import math
import os
import json
def draw_polygon(img, points, color, thickness=1):
    coordinates = points + (points[0],)

    for i, pt2 in enumerate(coordinates[1:]):
        pt1 = coordinates[i]
        cv2.line(img, pt1, pt2, color, thickness=int(thickness))

def rotate(point, degrees, origin):
    radians = math.radians(degrees)
    x, y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return round(qx), round(qy)


def calculate_all_points(coordinates):
    """
    Calculate all 4 points of provided coordinates
    :return: Tuple of tuples, one for each point.
    """
    p1, p3, rotation = coordinates
    # Un-rotate the point
    rp3 = rotate(p3, -rotation, p1)
    # Calculate the width
    w = rp3[0] - p1[0]
    h = rp3[1] - p1[1]
    # Calculate the new points (un-rotated)
    rp2 = (p1[0] + w, p1[1])
    rp4 = (p1[0], p1[1] + h)
    # Rotate the points
    p2 = rotate(rp2, rotation, p1)
    p4 = rotate(rp4, rotation, p1)
    points = (p1, p2, p3, p4)
    points = tuple(tuple(int(x) for x in t) for t in points)

    return points


def save_prediction_results_crf(model,graph, data, output_folder, image_folder):
    print('output_json_folder :', 'json')
    files = []
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
    aucs = []
    event_rates = []
    number_of_table_words = []
    number_of_nontable_words = []
    output_image_folder=PREDICTION_IMAGES_RESULT_PATH
    output_json_folder=PREDICTION_JSON_RESULT_PATH

    if not os.path.isdir(output_image_folder):
        os.mkdir(output_image_folder)
    if not os.path.isdir(output_json_folder):
        os.mkdir(output_json_folder)
    # f = open(output_folder + "/metrics.log", "w")
#   test_images = data[Constants.IMAGE_KEY].unique()
    best_images_folder = output_folder + "best_images/"
    medium_images_folder = output_folder + "medium_images/"
    good_images_folder = output_folder + "good_images/"
    bad_images_folder = output_folder + "bad_images/"
    all_images_folder = output_folder + "all_images/"

    if not os.path.isdir(best_images_folder):
        os.mkdir(best_images_folder)
    if not os.path.isdir(medium_images_folder):
        os.mkdir(medium_images_folder)

    if not os.path.isdir(good_images_folder):
        os.mkdir(good_images_folder)
    if not os.path.isdir(bad_images_folder):
        os.mkdir(bad_images_folder)
    if not os.path.isdir(all_images_folder):
        os.mkdir(all_images_folder)

    for test_image, nodes, edges in graph:
        try:
            jpg_ = test_image + ".jpg"
            print('output_json_folder :',output_json_folder+jpg_[:-3]+'json')
            folder_jpg_ = image_folder + jpg_
            # print(folder_jpg_)
            image = cv2.imread(folder_jpg_)
            # print(image.shape)
        except FileNotFoundError:
            print("image not found")
            continue
        # table_coordinates = eval(gt_df[gt_df[Constants.IMAGE_KEY] == test_image]["Tables"].values[0])
        # for item in table_coordinates:
        #     points = calculate_all_points(((int(item[1]), int(item[0])),
        #                                    (int(item[3]), int(item[2])), 0))
        #     draw_polygon(image, points, (255, 0, 255), thickness=4)
        coords = (data[data[Constants.IMAGE_KEY] == test_image]["Coordinates"]).tolist()
        # gt_val = data[data[Constants.IMAGE_KEY] == test_image]["Is_Table_Word"]
        # gt = gt_val.tolist()
        preds = model.predict([(nodes, edges)])[0]
        #nodes_ is coordinates
        # print(preds)
        # print('coords',coords)
        # print(len(preds),len(coords))
        result={}
        enum=0
        print('len c',len(coords))
        for c, p in zip(coords, preds):

            fcoord = c[1:-1].split(", ")


            points = calculate_all_points(((int(fcoord[1]), int(fcoord[0])),
                                       (int(fcoord[3]), int(fcoord[2])), 0))
            # patch_coord=[int(fcoord[0]), int(fcoord[1]),int(fcoord[2]), int(fcoord[3])]
            result[c]=str(p)
            if p == 1:
                draw_polygon(image, points, (0, 255, 0), thickness=4)
            else:
                draw_polygon(image, points, (255, 0, 0), thickness=4)
            print(enum,c,type(c),p,type(p))
            enum+=1
        cv2.imwrite(output_image_folder+jpg_,image)
        with open(output_json_folder+jpg_[:-3]+'json','w') as fw:
            json.dump(result,fw)



        #
        # accuracy = accuracy_score(y_true=gt_val, y_pred=preds)
        # precision = precision_score(y_true=gt_val, y_pred=preds)
        # f1score = f1_score(y_true=gt_val, y_pred=preds)
        # recall = recall_score(y_true=gt_val, y_pred=preds)
        # try:
        #     auc = roc_auc_score(y_true=gt_val, y_score=preds)
        # except ValueError:
        #     auc = 0
        #     print("Image contains only one label", test_image)
        #
        # number_of_table_words.append(np.sum(gt_val.values))
        # number_of_nontable_words.append(len(gt) - number_of_table_words[-1])
        # event_rate = number_of_table_words[-1] / len(gt)
        #
        # files.append(test_image)
        # accuracies.append(accuracy)
        # precisions.append(precision)
        # f1scores.append(f1score)
        # recalls.append(recall)
        # event_rates.append(event_rate)
        # aucs.append(auc)
        #
        # f.write(str(test_image) + "\n")
        # f.write("Accuracy : " + str(accuracy_score(y_true=gt_val, y_pred=preds)) + "\n")
        # f.write("Precision : " + str(precision) + "\n")
        # f.write("Recall : " + str(recall) + "\n")
        # f.write("F1 Score : " + str(f1score) + "\n")
        # f.write("AUC : " + str(auc) + "\n")
        #
        # f.write("*******************************************************\n")
        #
        # for c, p, g in zip(coords, preds, gt):
        #
        #     fcoord = c[1:-1].split(", ")
        #
        #     points = calculate_all_points(((int(fcoord[1]), int(fcoord[0])),
        #                                    (int(fcoord[3]), int(fcoord[2])), 0))
        #
        #     if p == 1:
        #         draw_polygon(image, points, (0, 255, 0), thickness=4)
        #     else:
        #         draw_polygon(image, points, (255, 0, 0), thickness=4)
        # print(image.shape)
        # if auc > 0.98:
        #     cv2.imwrite(best_images_folder + test_image + ".jpg", image)
        # elif auc > 0.95:
        #     cv2.imwrite(good_images_folder + test_image + ".jpg", image)
        # elif auc < 0.60:
        #     cv2.imwrite(bad_images_folder + test_image + ".jpg", image)
        # else:
        #     cv2.imwrite(medium_images_folder + test_image + ".jpg", image)
        # cv2.imwrite(all_images_folder + test_image + ".jpg", image)
    # temp_dict = {"Filename": files, "Accuracy": accuracies, "Precision": precisions, "F1 score": f1scores,
    #              "Recall": recalls, "Event rate": event_rates, "No of table words": number_of_nontable_words,
    #              "No of nontable words": number_of_nontable_words, "AuC": aucs}
    #
    # df = pd.DataFrame(temp_dict)
    # df.to_csv(output_folder + "result.csv", index=False)



def save_prediction_results__multi_label_crf(preds, test_images, data, gt_df,
                                             output_folder, label_column, image_folder):
    files = []
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
    aucs = []
    event_rates = []
    number_of_table_words = []
    number_of_nontable_words = []

    f = open(output_folder + "/metrics.log", "w")
    good_images_folder = output_folder + "good_images/"
    bad_images_folder = output_folder + "bad_images/"
    all_images_folder = output_folder + "all_images/"

    if not os.path.isdir(good_images_folder):
        os.mkdir(good_images_folder)
    if not os.path.isdir(bad_images_folder):
        os.mkdir(bad_images_folder)
    if not os.path.isdir(all_images_folder):
        os.mkdir(all_images_folder)

    for test_image in test_images:
        try:
            jpg_ = test_image + ".jpg"
            image = cv2.imread(image_folder + jpg_)
        except FileNotFoundError:
            print("image not found")
            continue
        table_coordinates = eval(gt_df[gt_df[Constants.IMAGE_KEY] == test_image]["Tables"].values[0])
        for item in table_coordinates:
            points = calculate_all_points(((int(item[1]), int(item[0])),
                                           (int(item[3]), int(item[2])), 0))
            draw_polygon(image, points, (255, 0, 255), thickness=4)
        coords = (data[data[Constants.IMAGE_KEY] == test_image]["Coordinates"]).tolist()
        gt_val = data[data[Constants.IMAGE_KEY] == test_image][label_column]
        gt = gt_val.tolist()

        accuracy = accuracy_score(y_true=gt_val, y_pred=preds)
        precision = precision_score(y_true=gt_val, y_pred=preds)
        f1score = f1_score(y_true=gt_val, y_pred=preds)
        recall = recall_score(y_true=gt_val, y_pred=preds)
        try:
            auc = roc_auc_score(y_true=gt_val, y_score=preds)
        except ValueError:
            auc = 0
            print("Image contains only one label", test_image)

        number_of_table_words.append(np.sum(gt_val.values))
        number_of_nontable_words.append(len(gt) - number_of_table_words[-1])
        event_rate = number_of_table_words[-1] / len(gt)

        files.append(test_image)
        accuracies.append(accuracy)
        precisions.append(precision)
        f1scores.append(f1score)
        recalls.append(recall)
        event_rates.append(event_rate)
        aucs.append(auc)

        f.write(str(test_image) + "\n")
        f.write("Accuracy : " + str(accuracy_score(y_true=gt_val, y_pred=preds)) + "\n")
        f.write("Precision : " + str(precision) + "\n")
        f.write("Recall : " + str(recall) + "\n")
        f.write("F1 Score : " + str(f1score) + "\n")
        f.write("AUC : " + str(auc) + "\n")

        f.write("*******************************************************\n")

        for c, p, g in zip(coords, preds, gt):

            fcoord = c[1:-1].split(", ")

            points = calculate_all_points(((int(fcoord[1]), int(fcoord[0])),
                                           (int(fcoord[3]), int(fcoord[2])), 0))

            if p == 1:
                draw_polygon(image, points, (0, 255, 0), thickness=4)
            elif p == 0:
                draw_polygon(image, points, (255, 0, 0), thickness=4)
            elif p == 2:
                draw_polygon(image, points, (0, 0, 255), thickness=4)
            elif p == 3:
                draw_polygon(image, points, (0, 255, 255), thickness=4)
            elif p == 4:
                draw_polygon(image, points, (255, 255, 0), thickness=4)
            elif p == 5:
                draw_polygon(image, points, (255, 0, 255), thickness=4)
            elif p == 6:
                draw_polygon(image, points, (0, 0, 0), thickness=4)

        if auc > 0.95:
            cv2.imwrite(good_images_folder + test_image + ".jpg", image)
        if auc < 0.60:
            cv2.imwrite(bad_images_folder + test_image + ".jpg", image)

        cv2.imwrite(all_images_folder + test_image + ".jpg", image)
    temp_dict = {"Filename": files, "Accuracy": accuracies, "Precision": precisions, "F1 score": f1scores,
                 "Recall": recalls, "Event rate": event_rates, "No of table words": number_of_nontable_words,
                 "No of nontable words": number_of_nontable_words, "AuC": aucs}

    df = pd.DataFrame(temp_dict)
    df.to_csv(output_folder + "result.csv", index=False)
