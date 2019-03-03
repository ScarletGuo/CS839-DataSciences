path_train_X = 'data/original/train/'
path_train_Y = 'data/labeled/train/'
path_test_X = 'data/original/test/'
path_test_Y = 'data/labeled/test/'


def test_model(models):
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

def load_X(path):
	X = get_candidates(path)
	X = augment_features(X)
	return

def load_Y(path)
	# dictionary: {doc_id: list of names}
	return

def augment_features(X):
	return X

def get_candidates():
	return

def train_model(m):
	if m == "svm":
		return svm.SVC(kernel='linear', C=1)
	elif m == "decision_tree":
		return
	else:
		return


if __name__ == "__main__":
	results, ev = test_model(['svm','decision_tree'])
