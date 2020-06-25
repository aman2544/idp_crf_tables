import os
import cv2

if __name__ == '__main__':
    path1='/Users/amandubey/Documents/RZT/Tables_test_set/crf_rf_4/after_post_processing/'
    path2='/Users/amandubey/Documents/RZT/Tables_test_set/crf_rf_4/op_images/'
    op_path='/Users/amandubey/Documents/RZT/Tables_test_set/crf_rf_4/untitled folder/'

    p1='_table_border.jpg'
    p2='_words.jpg'
    names=os.listdir(path1)
    for i in names:
        name=i.rsplit('.')[0]
        cv2.imwrite(op_path+name+p1,cv2.imread(path1+i))
        cv2.imwrite(op_path+name+p2,cv2.imread(path2+i))