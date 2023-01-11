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


def norm_feat(feat, kind= "normalize"):
    # import pdb; pdb.set_trace()
    feat = np.array(feat)

    if kind == "standardize":
        return (feat - feat.mean())/feat.std()
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
    # print("Model type: ", archType)
    return res_modeltype
  
def get_quants(x, n):
    """
    :param x:
    :param n:
    :return:
    """
    q = np.linspace(0, 1, n)
    return np.quantile(x, q)
