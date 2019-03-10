import numpy as np
from n_grams import find_ngram_features
from sklearn import linear_model, svm, tree, ensemble, datasets
from sklearn.model_selection import cross_val_predict, GridSearchCV, cross_val_score
from extract_gt import *
import pandas as pd
import graphviz 
import ipdb

func_list = [tree.DecisionTreeClassifier, \
            ensemble.RandomForestClassifier, \
            svm.SVC, \
            linear_model.LinearRegression, \
            linear_model.LogisticRegression]

#{'max_depth':20, 'n_estimators':30, 'criterion':"entropy", 'max_features':None, 'min_samples_split':17}
func_param = [{'max_depth':9}, {}, {'kernel':'rbf', 'C':2, 'gamma': 0.1}, {}, {'C':1e5}]

path_train_X = 'data/original/train/'
path_train_Y = 'data/labeled/train/'
path_test_X = 'data/original/test/'
path_test_Y = 'data/labeled/test/'
"""
# demo
path_train_X = 'data/original/t/'
path_train_Y = 'data/labeled/t/'
path_test_X = 'data/original/t/'
path_test_Y = 'data/labeled/t/'
"""

def load_data(path_X, path_Y):
    X = load_X(path_X)
    X, Y = load_Y(path_Y, X)
    # X, Y are data frames
    # index for X: (ngram, doc_id)
    # index for Y: ngram
    return X, Y


def load_X(path):
    return find_ngram_features(path)


def load_Y(path, X):
    # dictionary: {doc_id: list of names}
    gt = extract_gt(path)
    y = [group['ngram'].apply(lambda x: x in gt[doc_id]) 
    for doc_id, group in X.groupby('doc_id')]
    Y = pd.concat(y).to_frame()
    Y.columns = ['gt']
    Y['ngram'] = X['ngram']
    return X.set_index(['ngram', 'doc_id']), Y.set_index(['ngram'])


def train_model(X, Y):
    """
    train classifier within function list, using Cross-Validation, find best F1 and best classifier
    """
    best_F1 = -1

    for idx in range(len(func_list)):
        func = func_list[idx]
        classifier = func(**func_param[idx])
        print("Using {}. {}".format(idx, func.__name__))
        
        # CV to get the generalization error of classifier
        Y_pred = cross_val_predict(classifier, X, Y, cv=5)
        F1 = calculate_PR(Y, Y_pred)
        #scores = cross_val_score(classifier, X, Y, cv=5)  # not use; should search classifier with highest F1
        #F1 = scores.mean()

        classifier.fit(X, Y)  # must call .fit() before call .predict()
        print("best_param = {}".format(classifier))
        
        if F1 > best_F1:
            best_idx = idx
            best_F1 = F1
            best_classifier = classifier

    print("Training finished: the best classifier is {}. Its best F1 score is {}.\n".format(best_classifier, best_F1))

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
    

def debug():
    # train
    X, Y = load_data(path_train_X, path_train_Y)
#     np.savetxt("train_X.csv", X, delimiter=",")
#     np.savetxt("train_Y.csv", Y, delimiter=",")
    X.to_csv("train_X.csv")
    Y.to_csv("train_Y.csv")
    test_X, test_Y = load_data(path_test_X, path_test_Y)
#     np.savetxt("test_X.csv", test_X, delimiter=",")
#     np.savetxt("test_Y.csv", test_Y, delimiter=",")
    X.to_csv("test_X.csv")
    Y.to_csv("test_Y.csv")
    train_X = X.values
    train_Y = Y.values
    best_classifier = train_model(train_X, train_Y)

    # debug
    debug_X = X.values # load_X(path_debug_X)
    debug_Y = Y.values # load_Y(path_debug_Y)
    debug_pred = debug_PQ_set(debug_X, debug_Y, best_classifier)
    return pd.DataFrame(data=np.hstack[Y.index.values,debug_pred,debug_Y], columns=['ngram','predict', 'gt'])

    # test
    
    #test_model(test_X, test_Y, best_classifier)


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
    X, Y = load_data(path_train_X, path_train_Y)
    
    train_size = 2500
    train_X = X.values  # exclude n_grams and file_id
    train_Y = np.squeeze(Y.values) # exclude n_grams
    best_classifier = train_model(train_X, train_Y)

    # debug
    debug_X = X.values  # exclude n_grams and file_id
    debug_Y = np.squeeze(Y.values) # exclude n_grams
    debug_PQ_set(debug_X, debug_Y, best_classifier)
    
    # test
    test_X, test_Y = load_data(path_test_X, path_test_Y)
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
