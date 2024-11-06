import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models,transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torch.optim import Adam

import gc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import copy
from captum.attr import Saliency, IntegratedGradients
from captum.attr import visualization as viz
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import quantus
import time
import os
import sys
from tqdm import tqdm

import attributions
from modify_relu import replace_relu_with_modifiedrelu
from hessian_idg import HessianIntegratedDirectionalGradients

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def Hessian_metrics(model, test_loader, threshold):
    model.eval()
    idg = HessianIntegratedDirectionalGradients(model)
    sparse = quantus.Sparseness()
    ROAD = quantus.ROAD()
    model = model.to(device)
    score_sparse = []
    score_ROAD = []
    x_batches = []
    y_batches = []
    a_batches = []
    count = 0
    for i, (x_batch, y_batch) in enumerate(test_loader):
        if count > 10:
            break
        count += len(x_batch)
        print("INFO: " + str(count))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        scores, indices = torch.max(output, 1)
        images = x_batch
        x_batch = x_batch.to(device).cpu().detach().numpy()
        y_batch = indices.to(device).cpu().numpy()
        a_batch = np.array(idg.attribute(images, threshold=threshold), dtype=object)
        if np.all((a_batch == 0)):
            continue
        x_batch = x_batch.astype(np.float64)
        a_batch = a_batch.astype(np.float64)
        x_batches.extend(x_batch)
        y_batches.extend(y_batch)
        a_batches.extend(a_batch)
        try:
            answer = sparse(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch.sum(axis=1), device=device)
            score_sparse.extend(answer)
            answer = ROAD(model=model, x_batch=x_batch, y_batch=y_batch, a_batch=a_batch, device=device)
            score_ROAD.append(answer)
        except:
            pass
    np.save("x_batches_resnet50", x_batches)
    np.save("y_batches_resnet50", y_batches)
    np.save("a_batches_resnet50", a_batches)
    return score_sparse, score_ROAD


if __name__ == "__main__":
    data_transforms = {
        'validation':
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ]),
    }
    
    image_datasets = {
        'validation': 
        datasets.ImageFolder('data/imagenette2/val', data_transforms['validation'])
    }
    
    dataloaders = {
        'validation':
        torch.utils.data.DataLoader(image_datasets['validation'],
                                    batch_size=1,
                                    shuffle=False) 
    }
    
    model = models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load('model/resnet50.pth'))
    model = model.to(device)
    
    threshold = 0.50
    sparse, ROAD = Hessian_metrics(model, dataloaders["validation"], threshold)
    np.save("sparse_resnet50", sparse)
    np.save("ROAD_resnet50", ROAD)
    print(np.nanmean(sparse), np.nanstd(sparse))
