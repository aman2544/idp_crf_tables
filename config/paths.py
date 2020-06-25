import os
# ROOT_PATH = "/Users/sunilkumar/concept_learning/image_classification/"
#
# Z_DIM = 2
#
# EXPERIMENT_ID = Z_DIM
#
# BATCH_SIZE = 64
# BASE_PATH = os.path.join(ROOT_PATH,"Exp_{:02d}/".format(EXPERIMENT_ID))
# BASE_PATH=ROOT_PATH+'input_folder/'
# MODEL_NAME ="VAE"
# SPLIT_NAME="Split_1"
#
# WORD_FEATURE_FILENAME = "5e9eaf35b586ad3a142f57d7_0_Features.tsv"
# #WORD_FEATURE_FILENAME = "Features.tsv"
#
# # PREDICTION_RESULTS_PATH = BASE_PATH+"prediction_results/"
# EVIDENCE_PATH = BASE_PATH + "input/images_english_positive/evidence/"
# DATASET_NAME = 'mnist'
# DATASET_ROOT_PATH = os.path.join(BASE_PATH,"datasets/")
# DATASET_PATH = os.path.join(DATASET_ROOT_PATH, DATASET_NAME+"/")
# DATASET_PATH=BASE_PATH


# PREDICTION_RESULTS_PATH_DATASET = DATASET_PATH +MODEL_NAME+"/prediction_results/"
# PREDICTION_RESULTS_PATH_DATASET=ROOT_PATH
# PREDICTION_RESULTS_PATH=ROOT_PATH
#IMAGE_PATH = BASE_PATH + "input/images_english_positive/resized_image/"
# IMAGE_PATH = DATASET_PATH + "image_folder/"
# DATASET_PATH_COMMON_TO_ALL_EXPERIMENTS = os.path.join(ROOT_PATH,"datasets/"+DATASET_NAME)
################################################################################################################################

MODEL_PATH='/Users/amandubey/Music/tables_crf/existing_files/'

ROOT_PATH= "/Users/amandubey/Documents/RZT/Tables_test_set/"

WORD_FEATURES_PATH = ROOT_PATH+'output_csv_folder/'

BASE_PATH=ROOT_PATH+'input_folder/'
IMAGE_PATH = BASE_PATH + "image_folder/"
WORD_IMAGE_PATH=BASE_PATH+'words_image/'
JSON_FOLDER=BASE_PATH+'evidence_folder/'

PREDICTION_RESULTS_PATH=ROOT_PATH+'crf_rf_4/'
PREDICTION_IMAGES_RESULT_PATH=PREDICTION_RESULTS_PATH+'op_images/'
PREDICTION_JSON_RESULT_PATH=PREDICTION_RESULTS_PATH+'word_predictions/'
POSTPROCESSING_RESULT_PATH=PREDICTION_RESULTS_PATH+'after_post_processing/'
