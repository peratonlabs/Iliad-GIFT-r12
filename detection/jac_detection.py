import os
import json
import torch
import pickle
import numpy as np
from utils import utils
from joblib import dump, load
from utils.abstract import AbstractDetector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression


class JACDetector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath, scale_parameters_filepath, arg_dict):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
     
        self.arg_dict = arg_dict
        self.metaparameter_filepath = metaparameter_filepath
        # self.scale_parameters_filepath = scale_parameters_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath


    def automatic_configure(self, models_dirpath):
        #TODO: will implement this method later
        return 

    
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
        feats = get_jac_feats(model_filepath, nsamples=metaparameters["train_nsamples"])

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

        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)

        num_cv_trials = self.arg_dict['num_cv_trials']
        metaparameters = json.load(open(self.metaparameter_filepath, "r"))

        modelList = sorted(os.listdir(models_dirpath))
        # model_filepaths = [os.path.join(models_dirpath, modeldir, 'model.pt') for modeldir in modeldirs]
        
        # 

        x = []
        y = []

        jac_dets= str(metaparameters)

        # import pdb;pdb.set_trace()
        scratch_dirpath = self.arg_dict["scratch_dirpath"]
        holdoutratio = metaparameters["train_holdoutratio"]

        results_dir = os.path.join(scratch_dirpath, 'jac_results')
        os.makedirs(results_dir, exist_ok=True)


        # if scratch is not None and os.path.exists(scratch):
        #     with open(scratch,'rb') as f:
        #         xsv = pickle.load(f)
        # else:
        #     xsv = {}
        for model_id in modelList:
            print("Current model: ", model_id)
            model_result = None
            curr_model_dirpath = os.path.join(models_dirpath, model_id)

                
                
            res_path = os.path.join(results_dir, model_id + '.p')
            if os.path.exists(res_path):
                with open(res_path, 'rb') as f:
                    saved_model_result = pickle.load(f)
                model_result = saved_model_result
                print("loading saved")
            # import pdb; pdb.set_trace()
            if model_result is None:
                print('getting feats from', model_id)
                feats = get_jac_feats(os.path.join(curr_model_dirpath, 'model.pt'), nsamples=metaparameters["train_nsamples"])
                cls = utils.get_class_r12(os.path.join(curr_model_dirpath, 'config.json'))

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


        numSample = num_cv_trials
        for _ in range(numSample):
            ind = np.arange(len(y))
            np.random.shuffle(ind)

            split = round(len(y) * (1-holdoutratio))

            xtr = x[ind[:split]]
            xv = x[ind[split:]]
            ytr = y[ind[:split]]
            yv = y[ind[split:]]

            model = RandomForestClassifier(n_estimators=metaparameters["train_random_forest_regressor_param_n_estimators"])
            model.fit(xtr, ytr)
            pv = model.predict_proba(xv)
            pv = pv[:, 1]

            vroc = roc_auc_score(yv, pv)
            vkld = log_loss(yv, pv)
            
            rf_scores.append(pv)
            truths.append(yv)
            rocList.append(vroc)

            print('val auc:', vroc, 'pre-cal ce:', vkld)
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
            p2tst = np.clip(p2tst, 0.01, 0.99)
            ISOce_scores.append(log_loss(ytst, p2tst))
        print('post-cal ce (ISO): ', np.mean(ISOce_scores))

        rf_model = RandomForestClassifier(n_estimators=metaparameters["train_random_forest_regressor_param_n_estimators"])
        rf_model.fit(x, y)
        rf_scores = np.concatenate(rf_scores)
        rf_sample_y = np.concatenate(truths)
        ir_model = IsotonicRegression(out_of_bounds='clip')
        ir_model.fit(rf_scores, rf_sample_y)
        dump(rf_model, os.path.join(self.arg_dict['learned_parameters_dirpath'], 'cv_rf.joblib'))
        dump(ir_model, os.path.join(self.arg_dict['learned_parameters_dirpath'], 'cv_ir.joblib'))







def get_jac_feats(model_filepath, nsamples=1000, input_scale=1.0):
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_filepath)
    model.parameters()
    model.cuda()
    model.train()
    input_sz = model.parameters().__next__().shape[1]
    inputs = input_scale*torch.randn([nsamples,1,input_sz],device=device)
    jacobian = utils.compute_jacobian(model, inputs)
    return jacobian.mean(axis=1).reshape(-1)
