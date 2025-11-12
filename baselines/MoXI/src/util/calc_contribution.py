import torch
import numpy as np
import time
import torch.nn.functional as F
from .calc_reward import get_reward_shapley, get_reward_interaction

def choose_patch(curve_method, reward, identified_patch):
    if curve_method == 'insertion':
        reward[identified_patch] = -np.inf
        index = torch.argmax(reward).item()
    elif curve_method == 'deletion':
        reward[identified_patch] = np.inf
        index = torch.argmin(reward).item()
    return index


def online_identifying(model, image, label):
    # self context shapley
    start_time = time.time()
    reward = get_reward_shapley('insertion', model, image, label)
    print(f"First reward Calc: {time.time() - start_time}")
    # choose first patch
    start_time = time.time()
    identified_patch = [choose_patch('insertion', reward, identified_patch=[])]
    print(f"First patch: {time.time() - start_time}")
    # calc interaction
    
    for _ in range(10**2-1):
        start_time = time.time()
        
        reward = get_reward_interaction('insertion', model, image, label, identified_patch) #!!
        new_patch = choose_patch('insertion', reward, identified_patch) #!!
        identified_patch.append(new_patch)
#         print(f"Recursive Patch: {time.time() - start_time}")
    return identified_patch