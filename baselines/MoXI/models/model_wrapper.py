import torch
import torch.nn.functional as F
import numpy as np
import copy

device = torch.device('cuda:0')

# When using the mask method with a replace baseline value, calculate the logits of the model.
def model_wrapper_replace_baselinevalue(model, image, mask_list):
    B, C, H, W = image.shape
    M = len(mask_list)
    masked_list = np.zeros((M, C, 10 ** 2))
    for index in range(len(mask_list)):
        masked_list[index, :, mask_list[index]] = 1
    masked_list = masked_list.reshape(M, 3, 10, 10)
    mask = F.interpolate(torch.tensor(masked_list).clone(), size=[H, W], mode='nearest').float().to(device) # 4, 3, 224, 224
    masked_inputs = copy.deepcopy(image).to(device)
    masked_inputs = image * mask
    with torch.no_grad():
        output = model(masked_inputs)
    return output