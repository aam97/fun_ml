#Scikit-learn
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.utils import class_weight
#from sklearn.externals import joblib
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV


#Scipy
from scipy import stats
import numpy as np
import numpy.lib.recfunctions as recfn

def load_npy(fileName, isSig):

    data = np.load(fileName)
    if isSig:
        label = np.ones(len(data))
    else:
        label = np.zeros(len(data))

    data_labeled = recfn.rec_append_fields(data, 'label', label)

    return data_labeled

def split(data, test_size, cols):

    data_train, data_test, label_train, label_test = train_test_split(data, data['label'], test_size=test_size, random_state=0)

    X_train = data_train[cols]
    X_test = data_test[cols]

    y_train = label_train
    y_test = label_test

    X_train = X_train.astype([(name, '<f8') for name in X_train.dtype.names]).view(('<f8', len(X_train.dtype.names))) #convert from recarray to normal array
    X_test = X_test.astype([(name, '<f8') for name in X_test.dtype.names]).view(('<f8', len(X_test.dtype.names))) #convert from recarray to normal array

    return X_train, X_test, y_train, y_test

