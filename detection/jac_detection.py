import os
import json
import torch
import logging
import pickle
import numpy as np
from utils import utils
from joblib import dump, load
from utils.abstract import AbstractDetector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import hpsearch



logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    )
class JACDetector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath, arg_dict):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        logging.info("Initializing the detector class")
     
        self.arg_dict = arg_dict
        self.metaparameter_filepath = metaparameter_filepath
        self.scale_parameters_filepath = scale_parameters_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.default_metaparameters = json.load(open(metaparameter_filepath, "r"))



    def write_metaparameters(self, metaparameters):
        logging.info("Writing metaparameter to file")
        with open(self.learned_parameters_dirpath, "w") as fp:
            json.dump(metaparameters, fp)





    def automatic_configure(self, models_dirpath):

        """
        A function to automatically re-configure the detector by performing a grid search on a preset range of meta-parameters. 
        This function should automatically change the meta-parameters, call manual_configure and output a new meta-parameters.json 
        file (in the learned_parameters folder) when optimal meta-parameters are found.

        """
        logging.info("Configuring hyperparameters")

        scratch_dirpath = self.arg_dict["scratch_dirpath"]
        results_dir = os.path.join(scratch_dirpath, 'jac_results')

        if not os.path.exists(results_dir):
            self.manual_configure(models_dirpath)
        #LogisticRegression
        C_options =  list(range(1,28))
        opt_C = hpsearch.hpsearch_C(C_options, results_dir)
        modified_metaparameters = self.default_metaparameters.copy()
        modified_metaparameters["train_C"] = opt_C
        self.write_metaparameters(modified_metaparameters)
        #Randomforset
        # max_depth_options = range(1, 20)
        # opt_max_depth = hpsearch.hpsearch_max_depth(max_depth_options,results_dir)
        # modified_metaparameters = self.default_metaparameters.copy()
        # modified_metaparameters["train_max_depth"] = opt_max_depth
        # self.write_metaparameters(modified_metaparameters)

    



    
    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        metaparameters = json.load(open(self.metaparameter_filepath, "r"))
        basepath = self.arg_dict["gift_basepath"]
        feats_jac_rand = utils.get_jac_feats(model_filepath, nsamples=metaparameters["train_nsamples"])
        feats_ws  = utils.get_all_weights(model_filepath, basepath)
        feats_jac_real = utils.get_jac_feats_with_real_inputs(model_filepath,examples_dirpath, basepath)
        feats = np.concatenate([feats_jac_real, feats_jac_rand, feats_ws], axis = 0)
        rf_path = os.path.join(self.learned_parameters_dirpath, "cv_rf.joblib") 
        rf_path = os.path.join(basepath, rf_path)
        ir_path = os.path.join(self.learned_parameters_dirpath, "cv_ir.joblib") 
        ir_path = os.path.join(basepath, ir_path)

        rf_model = load(rf_path)
        ir_model = load(ir_path)
        pv = rf_model.predict_proba([feats])[:, 1]

        prob = ir_model.transform(pv)
        return prob


    def manual_configure(self, models_dirpath):
        """Configuration of the detector using the parameters from the metaparameters
            JSON file.
            Args:
                models_dirpath: str - Path to the list of model to use for training
        """
        logging.info("Runing manual configuration")
        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)


        num_cv_trials = self.arg_dict['num_cv_trials']
        metaparameters = json.load(open(self.metaparameter_filepath, "r"))
        modelList = sorted(os.listdir(models_dirpath))

        x = []
        y = []

        jac_dets= str(metaparameters)


        C = metaparameters["train_C"]

        # n_estimators = metaparameters["train_n_estimators"]
        # max_depth = metaparameters["train_max_depth"]
        # criterion = metaparameters["train_criterion"]

        scratch_dirpath = self.arg_dict["scratch_dirpath"]
        holdoutratio = metaparameters["train_holdoutratio"]
        basepath = self.arg_dict["gift_basepath"]

        results_dir = os.path.join(scratch_dirpath, 'jac_results')
        os.makedirs(results_dir, exist_ok=True)

        for model_id in modelList:

            model_result = None
            curr_model_dirpath = os.path.join(models_dirpath, model_id)

            example_dirpath = os.path.join(curr_model_dirpath, "clean-example-data")
            res_path = os.path.join(results_dir, model_id + '.p')
            if os.path.exists(res_path):
                with open(res_path, 'rb') as f:
                    saved_model_result = pickle.load(f)
                model_result = saved_model_result
                print("loading saved")
            if model_result is None:
                print('getting feats from', model_id)
                model_filepath = os.path.join(curr_model_dirpath, 'model.pt')
                feats_jac_rand = utils.get_jac_feats(model_filepath, nsamples=metaparameters["train_nsamples"])
                feats_jac_real = utils.get_jac_feats_with_real_inputs(model_filepath,example_dirpath, basepath)
                feats_ws = utils.get_all_weights(model_filepath)
                cls = utils.get_class_r12(os.path.join(curr_model_dirpath, 'config.json'))
                feats = np.concatenate([feats_jac_real, feats_jac_rand, feats_ws], axis = 0)

                model_result = {"jac_dets": jac_dets, 'cls': cls, 'features': feats}
                with open(res_path, "wb") as f:
                    pickle.dump(model_result, f)
            x.append(model_result["features"])
            y.append(model_result["cls"])

        x = np.stack(x, 0)
        y = np.array(y)
        rf_scores = []
        truths = []
        rocList = []
        vrocThreshold = 1
        numSample = num_cv_trials

        for _ in range(numSample):
            ind = np.arange(len(y))
            np.random.shuffle(ind)

            split = round(len(y) * (1-holdoutratio))
            xtr = x[ind[:split]]
            xv = x[ind[split:]]
            ytr = y[ind[:split]]
            yv = y[ind[split:]]

            # model = LogisticRegression(max_iter=1000, C=C)
            # model = RandomForestClassifier(n_estimators=1000)
            model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
            model.fit(xtr, ytr)
            pv = model.predict_proba(xv)
            pv = pv[:, 1]
            
    
            try:
                print("auc: ",roc_auc_score(yv, pv), " ce: ",log_loss(yv, pv))
                vroc = roc_auc_score(yv, pv)
                rocList.append(vroc)                      
                rf_scores.append(pv)
                truths.append(yv)
        
            except:
                print('AUC error (probably due to class balance)')
        print('avg auc: ', np.mean(rocList))

  

        rf_scores = np.array(rf_scores)
        truths = np.array(truths)
        
        ISOce_scores = []
        for _ in range(10):
            ind = np.arange(len(rf_scores))
            np.random.shuffle(ind)
            split = round(len(rf_scores) * (1-holdoutratio))

            ptr = np.concatenate(rf_scores[ind[:split]])
            ptst = np.concatenate(rf_scores[ind[split:]])
            ytr = np.concatenate(truths[ind[:split]])
            ytst = np.concatenate(truths[ind[split:]])


            ir_model = IsotonicRegression(out_of_bounds='clip')
            ir_model.fit(ptr, ytr)
            p2tst = ir_model.transform(ptst)
            ISOce_scores.append(log_loss(ytst, p2tst))
        print('post-cal ce (ISO): ', np.mean(ISOce_scores))

        
        # rf_model = LogisticRegression(max_iter=1000, C=C)
        # rf_model = RandomForestClassifier(n_estimators=1000)
        rf_model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
        rf_model.fit(x, y)
        rf_scores = np.concatenate(rf_scores)
        rf_sample_y = np.concatenate(truths)
        ir_model = IsotonicRegression(out_of_bounds='clip')
        ir_model.fit(rf_scores, rf_sample_y)
        dump(rf_model, os.path.join(self.arg_dict['learned_parameters_dirpath'], 'cv_rf.joblib'))
        dump(ir_model, os.path.join(self.arg_dict['learned_parameters_dirpath'], 'cv_ir.joblib'))
        logging.info("Training Done!")




