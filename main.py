import numpy as np
from n_grams import find_ngram_features
from sklearn import linear_model, svm, tree, ensemble, datasets
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score
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
            linear_model.LogisticRegression]
# linear_model.LinearRegression, \

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
        X = pd.read_csv(path_X, index_col=['ngram','doc_id', 'span'], true_values=['True'], false_values=['False'])
        Y = pd.read_csv(path_Y, index_col=0, true_values=['True'], false_values=['False'])
    else:
        X = load_X(path_X, workers)
        Y = load_Y(path_Y, X)
        Y.to_csv(label+'_Y.csv')
        X = bool_to_float(X.set_index(['ngram', 'doc_id', 'span']))
        X.to_csv(label+'_X.csv')
        # X, Y are data frames
        # index for X: (ngram, doc_id)
        # index for Y: ngram
    return X, Y


def load_X(path, workers):
    return find_ngram_features(path, workers)


def load_Y(path, X):
    # dictionary: {doc_id: list of names}
    gt = extract_gt(path)
    y = [group['ngram'].apply(lambda x: x in gt[doc_id]) 
    for doc_id, group in X.groupby('doc_id')]
    Y = pd.concat(y).to_frame()
    Y.columns = ['gt']
    Y['ngram'] = X['ngram']
    Y = Y.sort_index()
    return Y.set_index(['ngram'])


def train_model(X, Y):
    """
    train classifier within function list, using Cross-Validation, find best F1 and best classifier
    """
    best_F1 = -1
    classifiers = []

    for idx in range(len(func_list)):
        func = func_list[idx]
        classifier = func(**func_param[idx])
        print("Using {}. {}".format(idx, func.__name__))
        
        # CV to get the generalization error of classifier
#         Y_pred = cross_val_predict(classifier, X, Y, cv=5)
#         F1 = calculate_PR(Y, Y_pred)
        scores = cross_val_score(classifier, X, Y, scoring='f1', cv=5)  # not use; should search classifier with highest F1
        F1 = scores.mean()

        classifier.fit(X, Y)  # must call .fit() before call .predict()
        print("best_param = {}".format(classifier))
        
        classifiers.append(classifier)
        
        if F1 > best_F1:
            best_idx = idx
            best_F1 = F1
            best_classifier = classifier

    print("Training finished: the best classifier is {}. Its best F1 score is {}.\n".format(best_classifier, best_F1))

    return classifiers, best_classifier

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
def rule_based_post_processing(Y_pred):
    pass

def debug_PQ_set(X, Y, best_classifier):
    """
    Split training data into P and Q:
    train M on P, apply M to label examples in Q, 
    then identify and debug the false positive/negative examples; 
    if you try to improve recall then pay attention to the false negative examples
    """
    print("Try the best classifier on debug mode!\n")
    #print("Best classifier is {}".format(best_classifier.__name__))
    print("Best_param : {}".format(best_classifier))

    print("Before rule-based postprocessing step:\n")
    Y_pred = best_classifier.predict(X)
    F1 = calculate_PR(Y, Y_pred)

    print("After rule-based postprocessing step:\n")
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

def test_model(X, Y, best_classifier):
    """
    Now we have found the best one classifier, apply it onto the test data
    """
    print("Apply the best classifier on test mode!\n")
    #print("Best classifier is {}".format(best_classifier.__name__))
    print("Best_param : {}".format(best_classifier))
    
    print("Before rule-based postprocessing step:\n")
    Y_pred = best_classifier.predict(X)
    F1 = calculate_PR(Y, Y_pred)

    print("After rule-based postprocessing step:\n")
    #Y_pred_after = rule_based_post_processing(Y_pred)
    #calculate_PR(Y, Y_pred_after)
    
class NameIdentifier(object):
    
    def __init__(self, num_workers=4, from_csv=True, demo=False):
        if from_csv:
            self.X, self.Y = load_data("train_X.csv", "train_Y.csv", from_csv=True)
            self.test_X, self.test_Y = load_data("test_X.csv", "test_Y.csv", from_csv=True)
        else:
            path_train_X, path_train_Y, path_test_X, path_test_Y = NameIdentifier.get_path(demo)
            self.X, self.Y = load_data(path_train_X, path_train_Y, label='train', workers=num_workers)
            self.test_X, self.test_Y = load_data(path_test_X, path_test_Y, label='test', workers=num_workers)
        self.classifiers = []
        self.best_classifier = None
    
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
            
    def train(self):
        train_X = self.X.values
        train_Y = np.squeeze(self.Y.values)
        self.classifiers, self.best_classifier = train_model(train_X, train_Y)
    
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
        return df
        
    def test(self):
        debug_X = self.test_X.values # load_X(path_debug_X)
        debug_Y = np.squeeze(self.test_Y.values) # load_Y(path_debug_Y)
        debug_pred = debug_PQ_set(debug_X, debug_Y, self.best_classifier)
        return NameIdentifier.get_debug_df(self.test_X, self.test_Y, debug_pred, debug_Y)
    
    def debug(self, X, Y, classifier):
        debug_X = X.values # load_X(path_debug_X)
        debug_Y = np.squeeze(Y.values) # load_Y(path_debug_Y)
        debug_pred = debug_PQ_set(debug_X, debug_Y, classifier)
        return NameIdentifier.get_debug_df(X, Y, debug_pred, debug_Y)


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
