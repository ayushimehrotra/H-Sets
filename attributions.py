# All code in this file was created by Ayushi Mehrotra.

import numpy as np
import matplotlib.pyplot as plt 
import torch
from captum.attr import Saliency, IntegratedGradients
from captum.attr import visualization as viz

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def imshow_grad(input):
    inp = np.transpose(input[0], (1,2,0))
    plt.imshow(inp, plt.cm.hot)


def my_gradient(model, input):
    '''
        Verified by Captum results as well
        model - model in question
        input - input in question
    '''
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    input.requires_grad = True
    output = model(input)
    score, indices = torch.max(output, 1)
    # print(indices)
    score.backward()
    grad = torch.abs(input.grad)
    grad = (grad - grad.min())/(grad.max() - grad.min())
    return grad.cpu().detach().numpy()


def gradient_image(model, input):
    return my_gradient(model, input)*input.squeeze().cpu().detach().numpy()


def imshow_ig(input):
    inp = np.transpose(input[0], (1,2,0))
    plt.imshow(inp, plt.cm.hot_r)


def my_integrated_gradients(model, input, steps):
    baseline = 0*input
    scaled_inputs = [baseline + (float(i)/steps)*(input-baseline) for i in range(0, steps+1)]
    grads = []
    for inp in scaled_inputs:
        inp = inp.detach()
        grads.append(my_gradient(model, inp))
    avg_grads = np.average(grads, axis=0)
    int_grads = (input.squeeze().cpu().detach().numpy()-baseline.cpu().detach().numpy())*avg_grads 
    return int_grads


def saliency(model, input):
    saliency = Saliency(model)
    input = torch.Tensor(input).to(device)
    outputs = model(input)
    _, pred = torch.max(outputs, 1)
    attributions_grad = saliency.attribute(input, target=pred)
    return attributions_grad.squeeze().cpu().detach().numpy()


def integrated_gradients(model, input):
    ig = IntegratedGradients(model)
    input = torch.Tensor(input).to(device)
    outputs = model(input)
    _, pred = torch.max(outputs, 1)
    attributions_ig = ig.attribute(input, target=pred)
    return attributions_ig.squeeze().cpu().detach().numpy()


def visualize(input):
    _ = viz.visualize_image_attr(np.transpose(input, (1, 2, 0)), None, method='heat_map', show_colorbar=True, sign='positive', title='Attribution')
