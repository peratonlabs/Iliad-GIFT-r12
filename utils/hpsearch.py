import os
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss


def hpsearch_max_depth(max_depth_options, results_dir = None, holdoutratio = 0.1):
    """
    :param results_dir:
    :param max_depth_options:

    output: returns optimal depth
    """

    results_dir =[]

    x = []
    y = []

    assert results_dir != None, "results_dir is not specified"

    modelList = sorted(os.listdir(results_dir))

    for model_id in modelList:
        curr_model_res_path = os.path.join(results_dir, model_id)

            
        with open(curr_model_res_path, 'rb') as f:
            saved_model_result = pickle.load(f)
        model_result = saved_model_result
        x.append(model_result["features"])
        y.append(model_result["cls"])

    x = np.array(x)
    y = np.array(y)
    pred_auc = []
    pred_ce = []
    for max_depth in  max_depth_options:

        ind = np.arange(len(y))
        np.random.shuffle(ind)

        split = round(len(y) * (1-holdoutratio))

        xtr = x[ind[:split]]
        xv = x[ind[split:]]
        ytr = y[ind[:split]]
        yv = y[ind[split:]]

        model = RandomForestClassifier(n_estimators=1000, max_depth = max_depth)
        model.fit(xtr, ytr)
        pv = model.predict_proba(xv)
        pv = pv[:, 1]

        vroc = roc_auc_score(yv, pv)
        vkld = log_loss(yv, pv)
        
        pred_auc.append(vroc)
        pred_ce.append(vkld)

    opt_index = np.argmin(pred_ce)
    return max_depth_options[opt_index]

def hpsearch_C(C_options, results_dir = None, holdoutratio = 0.1):
    """
    :param results_dir:
    :param C_options:

    output: returns optimal depth
    """

    x = []
    y = []

    assert results_dir != None, "results_dir is not specified"
    modelList = sorted(os.listdir(results_dir))

    for model_id in modelList:
        curr_model_res_path = os.path.join(results_dir, model_id)

        with open(curr_model_res_path, 'rb') as f:
            saved_model_result = pickle.load(f)
        model_result = saved_model_result
        x.append(model_result["features"])
        y.append(model_result["cls"])

    x = np.array(x)
    y = np.array(y)
    pred_auc = []
    pred_ce = []
    for C in C_options:

        ind = np.arange(len(y))
        np.random.shuffle(ind)

        split = round(len(y) * (1-holdoutratio))

        xtr = x[ind[:split]]
        xv = x[ind[split:]]
        ytr = y[ind[:split]]
        yv = y[ind[split:]]

        model = LogisticRegression(max_iter=1000, C=C)
        model.fit(xtr, ytr)
        pv = model.predict_proba(xv)
        pv = pv[:, 1]

        vroc = roc_auc_score(yv, pv)
        vkld = log_loss(yv, pv)

        pred_auc.append(vroc)
        pred_ce.append(vkld)

    opt_index = np.argmin(pred_ce)
    return C_options[opt_index]