import torch
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append('../')
from MoXI.models.model_wrapper import model_wrapper_replace_baselinevalue

#---------------self shapley value-----------------
def get_reward_shapley(curve_method, model, inputs, label):
    if curve_method =='insertion':
        masked_list = [[k] for k in range(10 ** 2)]

    elif curve_method =='deletion':
        all_i = list(range(10 ** 2))
        masked_list = []
        for i in all_i:
            masked_list.append([x for x in all_i if x != i])
    
    with torch.no_grad():
        logits = model_wrapper_replace_baselinevalue(model, inputs, masked_list)
        reward = get_reward(logits, label)
    return reward

#---------------self interactions-----------------
def get_reward_interaction(curve_method, model, inputs, label, identified_patch):
    if curve_method =='insertion':
        masked_list = [[k]+identified_patch for k in range(10 ** 2) ]

    elif curve_method =='deletion':
        all_i = list(range(0, 10 ** 2))
        masked_list = []
        for i in all_i:
            remove_subset = [x for x in all_i if x not in [i]+identified_patch]
            masked_list.append(remove_subset)
        for identified_patch_index in identified_patch:
            masked_list[identified_patch_index].pop(0)

    with torch.no_grad():
        logits = model_wrapper_replace_baselinevalue(model, inputs, masked_list)
        reward = get_reward(logits, label)
    return reward


#---------------apply a reward function-----------------
def get_reward(logits, label):
    v = F.log_softmax(logits, dim=1)[:, label]
    return v