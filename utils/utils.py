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