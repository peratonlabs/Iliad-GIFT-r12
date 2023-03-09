import os
import json
import torch
import numpy as np
import scipy as sc
from numpy import linalg 

from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig


def read_json(truth_fn):
    with open(truth_fn) as f:
        jsonFile = json.load(f)
    return jsonFile


def read_truthfile(truth_fn):
    with open(truth_fn) as f:
        truth = json.load(f)
    lc_truth = {k.lower(): v for k,v in truth.items()}
    return lc_truth


CLS_DICT ={"clean":0, "poisoned":1}
def get_class_r12(truth_fn):

    truth = read_truthfile(truth_fn)
    return CLS_DICT[truth["model_type"]]


def compute_jacobian(model, image):
    """
    :param model:
    :param image:
    :return:
    """
    image.requires_grad = True

    output = model(image)
    out_shape = output.shape

    jacobian = []
    y_onehot = torch.zeros([image.shape[0], out_shape[2]], dtype=torch.long).cuda()
    one = torch.ones([image.shape[0]], dtype=torch.long).cuda()
    for label in range(out_shape[2]):
        y_onehot.zero_()
        y_onehot[:, label] = one
        curr_y_onehot = torch.reshape(y_onehot, out_shape)
        output.backward(curr_y_onehot, retain_graph=True)
        jacobian.append(image.grad.detach().cpu().numpy())
        image.grad.data.zero_()

    del y_onehot, one, output
    return np.stack(jacobian, axis=0)


def norm_feat(feat, kind= ""):
    # import pdb; pdb.set_trace()
    feat = np.array(feat)

    if kind == "standardize":
        return (feat - feat.mean())/feat.std()
    elif kind=="std":
        return feat/feat.std()
    elif kind == "normalize":
        return  (feat - feat.min())/(feat.max()- feat.min())
    else:
        return  feat/feat.max()


def get_layer_stats(x):
    return [np.mean(x), np.std(x), np.min(x), np.max(x), sc.stats.skew(x)]

def get_layer_stats_vectorized_mean(model_filepath, whichLayer = "firstLast"): 
    model = torch.load(model_filepath)
    if  whichLayer == "firstLast":
        params = [p.cpu().detach().numpy().reshape(-1)  for name, p in model.named_parameters() if "bias" not in name]
        params_sub = [params[0], params[-1]]
        res = np.array(list(map(get_layer_stats, params_sub)))
        return res.mean(axis = 0)
    elif  whichLayer == "justweights":
        params = [p.cpu().detach().numpy().reshape(-1)  for name, p in model.named_parameters() if "bias" not in name]
        res = np.array(list(map(get_layer_stats, params)))
        return res.mean(axis = 0)
    else:
        params = [p.cpu().detach().numpy().reshape(-1) for p in model.parameters()]
        res = np.array(list(map(get_layer_stats, params)))
    return res.mean(axis = 0)



def get_layer_stats_vectorized(model_filepath, whichLayer = "firstLast"): 
    archMap = {'Net2':0, 'Net3':1, 'Net4':2, 'Net5':3, 'Net6':4 , 'Net7':5 }
    model = torch.load(model_filepath)
    archName = str(type(model)).split(".")[1][:4]
    archType = archMap[archName]


    if  whichLayer == "firstLast":
        params = [p.cpu().detach().numpy().reshape(-1)  for name, p in model.named_parameters() if "bias" not in name]
        params_extracted = [params[0], params[-1]]
    elif  whichLayer == "justweights":
        params_extracted = [p.cpu().detach().numpy().reshape(-1)  for name, p in model.named_parameters() if "bias" not in name]
    elif whichLayer=="all":
        params_extracted = [p.cpu().detach().numpy().reshape(-1) for p in model.parameters()]
    else:
        raise ValueError("please specify which layers to use!")
    weight_stats = np.concatenate(np.array(list(map(get_layer_stats, params_extracted))), axis = 0)
    res_modeltype = np.concatenate([[archType], weight_stats], axis = 0)


  

def get_layer_weights_vectorized(model_filepath, whichLayer = "firstLast"): 
    archMap = {'Net2':0, 'Net3':1, 'Net4':2, 'Net5':3, 'Net6':4 , 'Net7':5 }
    model = torch.load(model_filepath)
    archName = str(type(model)).split(".")[1][:4]
    archType = archMap[archName]


    if  whichLayer == "firstLast":
        params = [p.cpu().detach().numpy().reshape(-1)  for name, p in model.named_parameters() if "bias" not in name]
        params_extracted = [params[0], params[-1]]
    elif  whichLayer == "justweights":
        params_extracted = [p.cpu().detach().numpy().reshape(-1)  for name, p in model.named_parameters() if "bias" not in name]
    elif whichLayer=="all":
        params_extracted = [p.cpu().detach().numpy().reshape(-1) for p in model.parameters()]
    else:
        raise ValueError("please specify which layers to use!")
    weight_stats = np.concatenate(np.array(list(map(get_layer_stats, params_extracted))), axis = 0)
    res_modeltype = np.concatenate([[archType], weight_stats], axis = 0)
    # print("Model type: ", archType)
    return res_modeltype

def get_weights_firstlayer(model_filepath, basepath="./"): 
    """
    :param model_filepath:
    """
    model = torch.load(model_filepath)

    params = [p.cpu().detach().numpy() for name, p in model.named_parameters() if "bias" not in name]
    input_size = params[0].shape[1]
    firstLayer =  [np.sort(params[0][:,i]) for i in  range(input_size)]
    firstLayer = np.concatenate(firstLayer)
    # firstLayer =  params[0].reshape(-1)
    feat = firstLayer.reshape(-1)
    return feat

def get_weights_firstlayer_and_lastlayer(model_filepath): 
    """
    :param model_filepath:
    """
    model = torch.load(model_filepath)
    params = [p.cpu().detach().numpy() for name, p in model.named_parameters() if "bias" not in name]
    bias = [p.cpu().detach().numpy() for name, p in model.named_parameters() if "bias" in name]
    input_size1 = params[0].shape[1]
    
    
    firstLayer =  [np.sort(params[0][:,i]) for i in  range(input_size1)]
    firstLayer = np.concatenate(firstLayer)  
    bias_first = bias[0]
    firstLayer = np.concatenate([firstLayer,bias_first ]) 

    bias_last = bias[-1]
    input_size2 = params[-1].shape[0]
    lastLayer = [np.sort(params[-1][i,:]) for i in range(input_size2)]
    lastLayer = np.concatenate(lastLayer)
    lastLayer = np.concatenate([lastLayer, bias_last])

    feats = np.concatenate([firstLayer, lastLayer])
    return feats


def get_all_weights(model_filepath, sort_first_layer=False): 
    """
    :param model_filepath:
    """
    model = torch.load(model_filepath)
    if sort_first_layer:
        sortWeight = lambda params:  [np.sort(params[:,i]) for i in  range(params.shape[1])]
        params = [p.cpu().detach().numpy().T for name, p in model.named_parameters() if "bias" not in name]
        params[0] = np.array(sortWeight(params[0])).T
    else:
        params = [p.cpu().detach().numpy().T for name, p in model.named_parameters() if "bias" not in name]

    feats = np.linalg.multi_dot(params)
    feats = feats.reshape(-1)
    stats_feats = get_layer_stats(feats)
    feats =  np.concatenate([feats, stats_feats], axis = 0)
    # import pdb; pdb.set_trace()

    return feats/feats.std()






def get_all_weights_with_reference(model_filepath, basepath = "./"): 
    """
    :param model_filepath:
    :param ref_path:
    """
    model = torch.load(model_filepath)
    numparams = len([p for p in model.parameters()])
    ref_path=os.path.join(basepath,"reference_models")

    ref_filepath = ref_path + "/"+str(type(model)).split(".")[1].split("'")[0] + "_"+str(numparams) +".pt"
    ref_model  = torch.load(ref_filepath)
    

    mod_params = [p.cpu().detach().numpy().T for name, p in model.named_parameters() if "bias" not in name]
    ref_params = [p.cpu().detach().numpy().T for name, p in ref_model.named_parameters() if "bias" not in name]
    params = [p1-p2 if p1.all()!=p1.all() else p1 for p1, p2 in zip(mod_params, ref_params) ]

    feats  = np.linalg.multi_dot(params)
    feats = feats.reshape(-1)
    
    return feats/feats.std()

    
def get_quants(x, n):
    """
    :param x:
    :param n:
    :return:
    """
    q = np.linspace(0, 1, n)
    return np.quantile(x, q)


def get_arch(model_filepath):
    model = torch.load(model_filepath)

    numparams = len([p for p in model.parameters()])
    cls = str(type(model)).split(".")[1][:4] 
    return cls


def get_jac_feats(model_filepath, nsamples=1000, input_scale=1.0):
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_filepath)
    model.parameters()
    model.cuda()
    model.train()
    input_sz = model.parameters().__next__().shape[1]
    inputs = input_scale*torch.randn([nsamples,1,input_sz],device=device)
    jacobian = compute_jacobian(model, inputs)
    # norm_const = linalg.norm(jacobian)
    # jacobian = jacobian.mean(axis=1).reshape(-1)
    # return jacobian/jacobian.std()
    # return jacobian
    jacobian = jacobian.mean(axis=1)
    dim_size = jacobian.shape[0]

    feat = np.concatenate([(jacobian[i]/jacobian[i].std()).reshape(-1) for i in range(dim_size)])

    return feat

def get_jac_feats_with_reference(model_filepath, nsamples=1000, input_scale=1.0,basepath="./"):
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_filepath)
    model.parameters()
    model.cuda()
    model.train()
    input_sz = model.parameters().__next__().shape[1]
    inputs = input_scale*torch.randn([nsamples,1,input_sz],device=device)
    jacobian = compute_jacobian(model, inputs)
    jacobian = jacobian.mean(axis=1).reshape(-1)

    numparams = len([p for p in model.parameters()])
    ref_path=os.path.join(basepath,"reference_models")


    ref_filepath = ref_path + "/"+str(type(model)).split(".")[1].split("'")[0] + "_"+str(numparams) +".pt"
    ref_model  = torch.load(ref_filepath)
    ref_model.parameters()
    ref_model.cuda()
    ref_model.train()
    input_sz = model.parameters().__next__().shape[1]
    inputs = input_scale*torch.randn([nsamples,1,input_sz],device=device)
    jacobian2 = compute_jacobian(ref_model, inputs)
    jacobian2 = jacobian2.mean(axis=1).reshape(-1)
    delta = jacobian-jacobian2
    return delta



def get_data_cyber(examples_dirpath, scale_params_path):
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     # Setup scaler
    scaler = StandardScaler()

    scale_params = np.load(scale_params_path)

    scaler.mean_ = scale_params[0]
    scaler.scale_ = scale_params[1]
    X = []
    for examples_dir_entry in os.scandir(examples_dirpath):
        if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
            feature_vector = np.load(examples_dir_entry.path).reshape(1, -1)
            feature_vector = torch.from_numpy(scaler.transform(feature_vector.astype(float))).float()
            X.append(feature_vector)
    inputs = torch.tensor(np.stack(X), device = device)
    return inputs

def get_jac_feats_with_real_inputs(model_filepath, examples_dirpath, scale_params_path):
    model = torch.load(model_filepath)
    model.parameters()
    model.cuda()
    model.train()
    
    inputs =get_data_cyber(examples_dirpath, scale_params_path)
    jacobian = compute_jacobian(model, inputs)

    jacobian = jacobian.mean(axis=1)
    dim_size = jacobian.shape[0]

    feat = np.concatenate([(jacobian[i]/jacobian[i].std()).reshape(-1) for i in range(dim_size)])

    return feat



def predict_proba_custom(score, threshold = 0.60, clip_lo=0.01, clip_hi=0.99):
    predict_proba= lambda s:  1 if s > threshold else 0
    predict_proba = np.vectorize(predict_proba)
    return np.clip(predict_proba(score), clip_lo, clip_hi)




