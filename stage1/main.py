import numpy as np
from n_grams import find_ngram_features
from sklearn import linear_model, svm, tree, ensemble, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score
from features import prefix
from extract_gt import *
import pandas as pd
import graphviz 
import logging
#import ipdb


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


func_list = [tree.DecisionTreeClassifier, \
             ensemble.RandomForestClassifier, \
             svm.SVC, \
             linear_model.LinearRegression, \
             linear_model.LogisticRegression]


#{'max_depth':20, 'n_estimators':30, 'criterion':"entropy", 'max_features':None, 'min_samples_split':17}
func_param = [{'max_depth':9}, {}, {'kernel':'rbf', 'C':2, 'gamma': 0.1}, {}, {'C':1e5}]


def bool_to_float(X):
    for c, t in zip(X.columns.values, X.dtypes):
        if c == "ngram":
            continue
        if t == 'object' or t == 'bool':
            X[c] = X[c].astype('float')
    return X


def load_data(path_X, path_Y, from_csv=False, label='train', workers=4):
    if from_csv:
        X = pd.read_csv(path_X, index_col=['ngram','doc_id', 'sts_id', 'span'], true_values=['True'], false_values=['False'])
        Y = pd.read_csv(path_Y, index_col=0, true_values=['True'], false_values=['False'])
    else:
        X = load_X(path_X, workers)
        X = bool_to_float(X.set_index(['ngram', 'doc_id', 'sts_id', 'span']))
        X.to_csv(label+'_X.csv')
        Y = load_Y(path_Y, X.reset_index())
        Y.to_csv(label+'_Y.csv')
        # X, Y are data frames
        # index for X: (ngram, doc_id)
        # index for Y: ngram
    return X, Y


def load_X(path, workers):
    return find_ngram_features(path, workers)


def load_Y(path, X):
    # dictionary: {doc_id: sts_id: list of names}
    gt = extract_gt(path)
    y = [group['ngram'].apply(lambda x: x in gt[cls[0]][cls[1]]) 
         for cls, group in X.groupby(['doc_id', 'sts_id'])]
    #y = [group['ngram'].apply(lambda x: x in gt[doc_id]) 
    #for doc_id, group in X.groupby('doc_id')]
    Y = pd.concat(y).to_frame()
    Y.columns = ['gt']
    Y['ngram'] = X['ngram']
    Y = Y.sort_index()
    return Y.set_index(['ngram'])


def train_model(X, Y):
    """
    train classifier within function list, using Cross-Validation, find best F1 and best classifier
    """
    X = get_x(X)
    Y = get_y(Y)
    best_score = -1

    for idx in range(len(func_list)):
        func = func_list[idx]
        classifier = func(**func_param[idx])
        print("Using {}. {}".format(idx, func.__name__))

        try:
            scores = cross_val_score(classifier, X, Y, scoring='f1', cv=5)  # not use; should search classifier with highest F1
            score = scores.mean()

            classifier.fit(X, Y)  # must call .fit() before call .predict()        

            if score > best_score:
                best_idx = idx
                best_score = score
                best_classifier = classifier
        except Exception as e:
            print("ERROR running {}: {}".format(classifier, e))

    print("Training finished: the best classifier is {}. Its best F1 score is {}.\n".format(best_classifier, best_score))

    return best_classifier

def calculate_PR(Y, Y_pred, thres=0.5):
    # calculate the Precision, Recall and F1 values, given prediction and gt label
    true_pred_num = 0
    total_pos_label_num = 0
    pred_pos_label_num = 0

    for i in range(len(Y)):
        # Binary classification
        if Y_pred[i] >= thres:
            Y_pred[i] = 1
        else:
            Y_pred[i] = 0
        # calculate
        if Y[i] == 1 and Y_pred[i] == 1:
            true_pred_num = true_pred_num + 1
        if Y[i] == 1:
            total_pos_label_num = total_pos_label_num + 1
        if Y_pred[i] == 1:
            pred_pos_label_num = pred_pos_label_num + 1
    try:
        assert (true_pred_num > 0),"true_pred_num = 0!"
        assert (pred_pos_label_num > 0),"pred_pos_label_num = 0!"
        P = float(true_pred_num)/pred_pos_label_num
        assert (total_pos_label_num > 0),"total_pos_label_num = 0!"
        R = float(true_pred_num)/total_pos_label_num
        F1 = (2 * P * R)/(P + R)
    
        print("- Precision(P) = {}/{} = {:.4f}".format(true_pred_num, pred_pos_label_num, P)) 
        print("- Recall(R) = {}/{} = {:.4f}".format(true_pred_num, total_pos_label_num, R)) 
        print("- F1 = {:.4f}\n".format(F1))
    
        return F1
    except:
        logging.warning('CANNOT Calculate F1')
        return 0

# post processing
def rule_based_post_processing(X, Y_pred):
    # one span only has one true, and prefer the longer one
    X_df = X.copy()
    X_df['predict'] = Y_pred
    for cls, group in X_df.reset_index().groupby(['span', 'doc_id', 'sts_id']):
        if group['predict'].sum() > 1:
            # pick the longest true one that does not contain a prefix
            candidates = group[group['predict']==1]
            idx = candidates.length.idxmax()
            X_df.loc[group.index.values, 'predict'] = 0
            X_df.loc[idx, 'predict'] = 1
    return X_df.predict.values
            

def debug_PQ_set(X, Y, best_classifier):
    """
    Split training data into P and Q:
    train M on P, apply M to label examples in Q, 
    then identify and debug the false positive/negative examples; 
    if you try to improve recall then pay attention to the false negative examples
    """
    X = get_x(X)
    Y = get_y(Y)
    #print("Before rule-based postprocessing step:\n")
    Y_pred = best_classifier.predict(X)
    F1 = calculate_PR(Y, Y_pred)

    # print("After rule-based postprocessing step:\n")
    #Y_pred_after = rule_based_post_processing(Y_pred)
    #calculate_PR(Y, Y_pred_after)

    #analyze_false_positive(Y, Y_pred_after)
    return Y_pred

def analyze_false_positive(Y, Y_pred):
    """
    Split training data into P and Q:
    train M on P, apply M to label examples in Q, 
    then identify and debug the false positive/negative examples; 
    if you try to improve recall then pay attention to the false negative examples
    """
    pass

def test_model(X, Y, best_classifier):
    """
    Now we have found the best one classifier, apply it onto the test data
    """
    
    print("Before rule-based postprocessing step:\n")
    XX = get_x(X)
    Y = get_y(Y)
    Y_pred = best_classifier.predict(XX)
    F1 = calculate_PR(Y, Y_pred)

    print("After rule-based postprocessing step:\n")
    Y_after = rule_based_post_processing(X.reset_index(), Y_pred)
    calculate_PR(Y, Y_after)
    
    return Y_pred, Y_after
    
def get_x(X):
    return X.values

def get_y(Y):
    return np.squeeze(Y.values)
    
class NameIdentifier(object):
    
    def __init__(self, num_workers=4, from_csv=True, demo=False):
        if from_csv:
            self.X, self.Y = load_data("train_X.csv", "train_Y.csv", from_csv=True)
            self.test_X, self.test_Y = load_data("test_X.csv", "test_Y.csv", from_csv=True)
        else:
            path_train_X, path_train_Y, path_test_X, path_test_Y = NameIdentifier.get_path(demo)
            self.X, self.Y = load_data(path_train_X, path_train_Y, label='train', workers=num_workers)
            self.test_X, self.test_Y = load_data(path_test_X, path_test_Y, label='test', workers=num_workers)
        self.training = None
        self.M = None
        
    def run(self):
        P, Q, P_label, Q_label = self.split_sets()
        self.first_cv(P, P_label)
        return self.debug(Q, Q_label)
        
    def split_sets(self):
        # split train into P and Q
        P, Q, P_label, Q_label =  train_test_split(self.X, self.Y, test_size=0.2)
        return P, Q, P_label, Q_label
    
    def first_cv(self, P, P_label):
        print("********** FIRST CV **********")
        self.M = train_model(P, P_label)
        
    def debug(self, X, Y):
        print("********** DEBUG **********")
        debug_pred = debug_PQ_set(X, Y, self.M)
        return NameIdentifier.get_debug_df(X, Y, debug_pred, get_y(Y))
    
    def test(self, classifier=None):
        if classifier is None:
            classifier = self.M
        print("********** TEST **********")
        pred, after = test_model(self.test_X, self.test_Y, classifier)
        return NameIdentifier.get_debug_df(self.test_X, self.test_Y, pred, get_y(self.test_Y)), NameIdentifier.get_debug_df(self.test_X, self.test_Y, after, get_y(self.test_Y))
    
    @staticmethod
    def get_path(demo):
        if not demo:
            path_train_X = 'data/original/train/'
            path_train_Y = 'data/labeled/train/'
            path_test_X = 'data/original/test/'
            path_test_Y = 'data/labeled/test/'
        else:
            path_train_X = 'data/original/t/'
            path_train_Y = 'data/labeled/t/'
            path_test_X = 'data/original/t/'
            path_test_Y = 'data/labeled/t/'
        return path_train_X, path_train_Y, path_test_X, path_test_Y
    
    @staticmethod
    def get_debug_df(X_df, Y_df, pred, Y):
        df = pd.DataFrame(data=np.hstack([Y_df.index.values.reshape(-1,1), 
                                          X_df.reset_index().doc_id.values.reshape(-1,1), 
                                          X_df.reset_index().span.values.reshape(-1,1), 
                                          pred.reshape(-1,1), 
                                          Y.reshape(-1,1)]), 
                                          columns=['ngram', 'doc_id', 'span', 'predict', 'gt']).astype(
                                          {'ngram': 'object', 'doc_id': 'int64', 'span': 'object',
                                           'predict': 'bool','gt':'bool'})
        return df[df['gt']!=df['predict']]
        
    


if __name__ == "__main__":
    # train
    from os.path import isfile
    """
    if isfile("train_X.csv") and isfile("train_Y.csv"):
        X = pd.read_csv("train_X.csv")
        Y = pd.read_csv("train_Y.csv")
    else:
        X, Y = load_data(path_train_X, path_train_Y)
        X.to_csv("train_X.csv")
        Y.to_csv("train_Y.csv")

    if isfile("test_X.csv") and isfile("test_Y.csv"):
        test_X = pd.read_csv("test_X.csv")
        test_Y = pd.read_csv("test_Y.csv")
    else:
        test_X, test_Y = load_data(path_test_X, path_test_Y)
        test_X.to_csv("test_X.csv")
        test_Y.to_csv("test_Y.csv")
    """
    path_train_X = 'data/original/train/'
    path_train_Y = 'data/labeled/train/'
    path_test_X = 'data/original/test/'
    path_test_Y = 'data/labeled/test/'
    X, Y = load_data(path_train_X, path_train_Y, label='train')
    
    train_size = 2500
    train_X = X.values  # exclude n_grams and file_id
    train_Y = np.squeeze(Y.values) # exclude n_grams
    best_classifier = train_model(train_X, train_Y)

    # debug
    debug_X = X.values  # exclude n_grams and file_id
    debug_Y = np.squeeze(Y.values) # exclude n_grams
    debug_PQ_set(debug_X, debug_Y, best_classifier)
    
    # test
    X, Y = load_data(path_test_X, path_test_Y, label='test')
    test_X = X.values
    test_Y = np.squeeze(Y.values)
    test_model(test_X, test_Y, best_classifier)


"""
    # train
    from sklearn.model_selection import ShuffleSplit
    results = []
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    train_X = load_X(path_train_X)
    train_Y = load_Y(path_train_Y)
    trained_models = {}
    for m in models:
        trained_models[m] = train_model(m)
        results.append([m, cross_val_score(trained_models[m], train_X, train_Y, cv=cv)])

    # evaluate
    evaluation = []
    test_X = load_X(path_test_X)
    test_Y = load_Y(path_test_Y)
    for m in trained_models:
        evaluation.append([m, evaluate(trained_models[m], test_X, test_Y)])
    return results, evaluation
"""
