import json
import torch
import numpy as np



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
    return [np.mean(x), np.std(x), np.min(x), np.max(x)]

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

def get_weights_firstlayer(model_filepath): 
    """
    :param model_filepath:
    """
    model = torch.load(model_filepath)
    params = [p.cpu().detach().numpy() for name, p in model.named_parameters() if "bias" not in name]
    input_size = params[0].shape[1]
    firstLayer =  [np.sort(params[0][:,i]) for i in  range(input_size)]
    firstLayer = np.concatenate(firstLayer)    
    return firstLayer

def get_weights_firstlayer_and_lastlayer(model_filepath): 
    """
    :param model_filepath:
    """
    model = torch.load(model_filepath)
    params = [p.cpu().detach().numpy() for name, p in model.named_parameters() if "bias" not in name]
    input_size1 = params[0].shape[1]
    
    firstLayer =  [np.sort(params[0][:,i]) for i in  range(input_size1)]
    firstLayer = np.concatenate(firstLayer)  

    input_size2 = params[-1].shape[0]
    lastLayer = [np.sort(params[-1][i,:]) for i in range(input_size2)]
    lastLayer = np.concatenate(lastLayer)

    feats = np.concatenate([firstLayer, lastLayer])
    return feats

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
    # import pdb; pdb.set_trace()
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
    return jacobian.mean(axis=1).reshape(-1)

def predict_proba_custom(score, threshold = 0.60):
    predict_proba= lambda s:  1 if s > threshold else 0
    predict_proba = np.vectorize(predict_proba)
    return np.clip(predict_proba(score), 0.01, 0.99)
