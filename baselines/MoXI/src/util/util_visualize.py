import cv2
import numpy as np
import torch
import copy
from pytorch_grad_cam.utils.image import show_cam_on_image
from MoXI.models.model_wrapper import model_wrapper_replace_baselinevalue
import math

def convert_image_to_heatmap(image, count_list, b=0, out_size=224, image_weight=0.5):
    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor of shape (B,C,H,W)")

    counts = torch.as_tensor(count_list).detach().cpu().numpy() if torch.is_tensor(count_list) \
             else np.asarray(count_list, dtype=np.float32)
    counts = counts.ravel()
    G = int(round(math.sqrt(len(counts))))
    if G * G != len(counts):
        raise ValueError(f"count_list length {len(counts)} is not a perfect square")
    mask = counts.reshape(G, G).astype(np.float32)
    m_rng = float(mask.max() - mask.min())
    mask = (mask - mask.min()) / (m_rng + 1e-8)
    mask_224 = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    return mask_224



def get_accuracy(model, image, label, insert_index_list, mask_num):
    outputs = model_wrapper_replace_baselinevalue(model, image, [insert_index_list[mask_num]])
    isCorrect = (torch.argmax(outputs, axis=1) == label).sum().item()
    confidence = torch.softmax(outputs, dim=1)[0][label].item()
    return isCorrect, confidence

def make_heatmap(identified_patch, model, image, label, curve_method):
    count = 0
    heatmap = [0 for _ in range(10 ** 2)]
    if curve_method == 'insertion':
        insert_list = [[identified_patch[i] for i in range(j)] for j in range(1, len(identified_patch)+1)]
        
        while True:
            isCorrect, confidence = get_accuracy(model, image, label, insert_list, count) #!!
            count += 1
            if isCorrect==True or count == 10 ** 2 :
                break
    elif curve_method == 'deletion':
        range_list = list(range(10 ** 2))
        deletion_list = []

        for identify_index in identified_patch:
            range_list.remove(identify_index)
            deletion_list.append(copy.deepcopy(range_list))
        while True:
            isCorrect, confidence = get_accuracy(model, image, label, deletion_list, count)
            count += 1
            if isCorrect==False or count == 10 ** 2:
                break

    heatmap_value = count
    for index in identified_patch:
        if heatmap_value == 0:
            break
        heatmap[index] += heatmap_value
        heatmap_value -= 1
    return heatmap, count