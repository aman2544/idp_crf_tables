import pandas as pd
# from config.paths import  WORD_FEATURES_PATH, WORD_FEATURE_FILENAME
from common.todo_fix_this_later import column_map,features
import numpy as np


WORD_FEATURE_FILENAME = "5e9eaf35b586ad3a142f57d7_0_Features.tsv"
# WORD_FEATURES_PATH = '/Users/amandubey/Documents/RZT/Tables_test_set/output_csv_folder/'

def load_word_features(word_features_path):
    df = pd.read_csv(word_features_path, sep="\t", index_col=False)
    # df = df.rename(columns=column_map)
    # df["Is_Table_Word"] = 0
    # df.loc[df["Table_Id"] > 0, "Is_Table_Word"] = 1
    return df


def load_word_features_test(dataset_column_name):
    df = pd.read_csv(WORD_FEATURES_PATH + WORD_FEATURE_FILENAME, sep="\t", index_col=False)
    df = df.rename(columns=column_map)
    df["Is_Table_Word"] = 0
    df.loc[df["Table_Id"] > 0, "Is_Table_Word"] = 1
    test_data = df[df[dataset_column_name] == 'test']
    test_x = test_data[features]
    test_y = test_data[["Is_Table_Word"]]
    return test_data, test_x, test_y


# Accept bas_path and dataset
def load_all_data(base_path, dataset,):
    fq_data_file_name = base_path + dataset + "/Features.tsv"

#    TODO train test split is not stratified
    data = pd.read_csv(fq_data_file_name, sep="\t")
    train_data = data[data["data_set_type"] == 'train']
    test_data = data[data["data_set_type"] == 'test']

    test_data.loc[test_data["Label"] > 0, "Label"] = 1
    train_data.loc[train_data["Label"] > 0, "Label"] = 1

    return train_data, test_data


def get_neighbouring_words(word, words):

    tlbr = eval(word)
    t, l, b, r = tlbr[0], tlbr[1], tlbr[2], tlbr[3]
    neighbours = []
    h = b - t
    box = [t - 2 * h, l - 20, b + 2 * h, r + 20]
    for w in words:
        if w == word:
            continue
        _tlbr = eval(w)
        _t, _l, _b, _r = _tlbr[0], _tlbr[1], _tlbr[2], _tlbr[3]
        if _b > box[0] and _t < box[2] and _l < box[3] and _r > box[1]:
            neighbours.append(w)
    return neighbours


def page_graphs(df, file_name_column, features,label_column="Is_Table_Word"):
    pages = df[file_name_column].unique()
    result = {}
    for i, page in enumerate(pages):
        print("Creating graph for page", i, page)
        words_df = df[df[file_name_column] == page]
        # y = words_df[label_column].values

        y=np.zeros((df.shape[0]))
        print('y is done')
        words = words_df["Coordinates"].values
        nodes = words_df[features].values
        graph = {}
        for word in words:
            neighbours = get_neighbouring_words(word, words)
            graph[word] = neighbours
        result[page] = [graph, nodes, words, y]
    return result
