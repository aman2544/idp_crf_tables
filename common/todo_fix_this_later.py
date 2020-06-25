from enum import Enum
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import os

#TODO manju to fill in rest
# class Constants(Enum):
#     IMAGE_KEY ="Filename"
NUM_DIGITS =3
MAX_ITER = 4000



column_map = { 'Filename':'Filename',
 'Text':'Text',
 'Coordinates':'Coordinates',
 'top_left_aligned':'Number_of_words_left_aligned_above',
'continous_top_left_aligned':'Number_of_words_left_aligned_above_contigous',
'continous_bottom_left_aligned':'Number_of_words_left_aligned_below_contigous',
'bottom_left_aligned':'Number_of_words_left_aligned_below',
 'top_right_aligned':'Number_of_words_right_aligned_above',
'continous_top_right_aligned':'Number_of_words_right_aligned_above_contigous',
'bottom_right_aligned':'Number_of_words_right_aligned_below',
'continous_bottom_right_aligned':'Number_of_words_right_aligned_below_contigous',
 'top_aligned':'Number_of_words_left_and_right_aligned_above',
'top_continous_aligned':'Number_of_words_left_and_right_aligned_above_contigous',
 'bottom_aligned':'Number_of_words_left_and_right_aligned_below',
 'bottom_continous_aligned':'Number_of_words_left_and_right_aligned_below_contigous',
'left_aligned':'Total_number_of_left_aligned_words',
 'left_continous_aligned':'Total_number_of_left_aligned_words_contigous',
 'right_aligned':'Total_number_of_right_aligned_words',
'right_continous_aligned':'Total_number_of_right_aligned_words_contigous',
 'fully_aligned':'Number_of_words_left_and_right_aligned_above_and_below',
 'fully_continous_aligned':'Number_of_words_left_and_right_aligned_above_and_below_contigous',
'left_alignment_score':'Left_alignment_score',
 'right_alignment_score':'Right_alignment_score',
'aggregate_alignment_score':'Total_score_for _all_words_above_and_below',
 'Area':'Total_whitespace_around_the_word',
 'Top_distance':'Whitespace_distance_above_normalized',
 'Bottom_distance':'Whitespace_distance_below_normalized',
'Left_Distance':'Whitespace_distance_to_the_left_of_normalized',
 'Right_Distance':'Whitespace_distance_to_the_right_of_normalized'}
 # 'Label':'Table_Id'}

_features_old_name = [
    "top_left_aligned",
    "continous_top_left_aligned",
    "bottom_left_aligned",
    "continous_bottom_left_aligned",
    "top_right_aligned",
    "continous_top_right_aligned",
    "bottom_right_aligned",
    "continous_bottom_right_aligned",
    "top_aligned",
    "top_continous_aligned",
    "bottom_aligned",
    "bottom_continous_aligned",
    "left_aligned",
    "left_continous_aligned",
    "right_aligned",
    "right_continous_aligned",
    "fully_aligned",
    "fully_continous_aligned",
    "left_alignment_score",
    "right_alignment_score",
    "aggregate_alignment_score",
    "Area",
    "Top_distance",
    "Bottom_distance",
    "Left_Distance",
    "Right_Distance",
]

features = [column_map[f] for f in column_map.keys() if f in _features_old_name]


def predict(models,x,y):
    column_headers =['ML Algorithm','Precision',"Recall","F1","AUC"]
    metrics_table = []
    metrics_table.append(column_headers)
    for model in models:
        preds= model.predict(x)

        precision = round(precision_score(y_true=y, y_pred=preds),NUM_DIGITS)
        recall =round(recall_score(y_true=y, y_pred=preds),NUM_DIGITS)
        f1 =  round(f1_score(y_true=y, y_pred=preds),NUM_DIGITS)
        auc =  round(roc_auc_score(y_true=y, y_score=preds),NUM_DIGITS)
        metrics_table.append([model.get_model_type(),precision,recall,f1,auc])
    return metrics_table


class Constants:
    IMAGE_KEY = "Filename"


class Model:
    def __init__(self, model_type, hyper_params=None):
        # TODO fix this
        self.model = None
        self.model_type = model_type
        self.hyper_params = hyper_params
    def save(self,path):
        if self.model is None:
            raise Exception("Uninitialized or untrained model. Please invoke intitialze or train method first")

        fully_qualified_filename = path + self.get_model_type() + ".sav"
        with open(fully_qualified_filename,"wb") as outfile:
            pickle.dump(self.model, outfile)

    def load(self,path):

        fully_qualified_filename = path + self.get_model_type() + ".sav"
        if not os.path.isfile(fully_qualified_filename):
            raise Exception("File does not exist",fully_qualified_filename)

        with open(fully_qualified_filename,"rb") as infile:
            self.model = pickle.load(infile)

    def predict(self, x):
        return self.model.predict(x)

    def get_model_type(self):
        _model_type = self.model_type
        if self.hyper_params is not None:
            for k, v in self.hyper_params.items():
                _model_type = _model_type + "_" + k + "_" + str(v)
        return _model_type

    def fit(self, x, y):
        if self.model_type == "svm":
            if self.hyper_params is not None:
                if "kernel" in self.hyper_params:
                    if self.hyper_params["kernel"] == "linear":
                        self.model = LinearSVC()
                    else:
                        self.model = SVC(kernel=self.model.hyper_params["kernel"])
                else:
                    self.model = SVC()
            else:
                self.model = SVC()
        elif self.model_type == "logreg":
            if self.hyper_params is not None:
                if "class_weight" in self.hyper_params:
                    self.model=LogisticRegression(max_iter=MAX_ITER,class_weight=self.hyper_params["class_weight"])
                else:
                    self.model = LogisticRegression(max_iter=MAX_ITER)
            else:
                self.model = LogisticRegression(max_iter=MAX_ITER)
        elif self.model_type == "rf":
            if self.hyper_params is not None:
                if "max_depth" in self.hyper_params:
                    self.model = RandomForestClassifier(max_depth=self.hyper_params["max_depth"])
                else:
                    self.model = RandomForestClassifier()
            else:
                self.model = RandomForestClassifier()
        elif self.model_type == "decision_tree":
            if self.hyper_params is not None:
                if "max_depth" in self.hyper_params:
                    self.model = DecisionTreeClassifier(max_depth=self.hyper_params["max_depth"])
                else:
                    self.model = DecisionTreeClassifier()
            else:
                self.model = DecisionTreeClassifier()
        elif self.model_type == "nb":
            self.model = GaussianNB()
        else:
            raise Exception("Unknown model")
        print("Start Training ", self.model_type)
        self.model.fit(x, y)
        print("Completed Training ", self.model_type)