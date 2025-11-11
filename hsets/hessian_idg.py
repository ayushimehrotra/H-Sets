import torch
import gc
import numpy as np
import copy
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import time
import math

import attributions
from modify_relu import replace_relu_with_modifiedrelu

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class HessianIntegratedDirectionalGradients:

    def __init__(self, model):
        self.model = model
        self.max_features = 2000
        self.ig = IntegratedGradients(model)

    def make_attribution(self, input, index, max_elements, threshold):
        '''
            Create interaction set with given maximum elements and threshold
        '''
        model_copy = copy.deepcopy(self.model)
        replace_relu_with_modifiedrelu(model_copy)
        current_attribution_image = np.zeros(input.squeeze().detach().cpu().numpy().shape)

        def compute_hessian(i, j, k, grad_output, input, max_elements, threshold):
            if np.count_nonzero(current_attribution_image) > max_elements:
                return

            input.requires_grad_(True)
            grad2rd = torch.autograd.grad(grad_output[0, i, j, k], input, retain_graph=True)[0]
            np_grad2rd = grad2rd.detach().cpu().numpy()
            np_grad2rd = (np_grad2rd - np_grad2rd.min()) / (np_grad2rd.max() - np_grad2rd.min() + 1e-8)  

            del grad2rd
            torch.cuda.empty_cache()

            np_grad2rd[np_grad2rd <= threshold] = 0
            # print("[INFO] Number of Features", np.count_nonzero(np_grad2rd), "with threshold", threshold)

            if np.count_nonzero(np_grad2rd) == 0:
                return

            flattened_arr = np_grad2rd.flatten()
            sorted_indices = np.argsort(flattened_arr)[::-1]

            bfs = []
            for flat_index in sorted_indices:
                indices = np.unravel_index(flat_index, np_grad2rd.shape)

                if np_grad2rd[indices] == 0:
                    break

                if current_attribution_image[indices[1]][indices[2]][indices[3]] == 0:
                    current_attribution_image[indices[1]][indices[2]][indices[3]] = 1
                    bfs.append(indices)
                    if np.count_nonzero(current_attribution_image) > max_elements:
                        return

            for indices in bfs:
                compute_hessian(indices[1], indices[2], indices[3], grad_output, input, max_elements, threshold)

        model_copy.eval()
        input.requires_grad_(True)

        output = model_copy(input)
        score, indices = torch.max(output, 1)
        grad_output = torch.autograd.grad(score, input, create_graph=True)[0]
        compute_hessian(index[1], index[2], index[3], grad_output, input, max_elements, threshold)

        del model_copy, input, output, score, indices, grad_output
        gc.collect()
        torch.cuda.empty_cache()

        return current_attribution_image

    def find_feature_interaction_sets_sam(self, input, k, threshold):
        '''
            Find the starting point for the feature interaction set
            using SAM and Integrated Gradients. Uses the maximum value
            from Integrated Gradients as a starting point.
        '''
        gc.collect()
        torch.cuda.empty_cache()

        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        sam.to(device="cuda")

        # mask_generator = SamAutomaticMaskGenerator(sam)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            min_mask_region_area=1000,
        )

        image = input.detach().cpu().numpy()
        image = np.transpose((image / 2) + 0.5, (1, 2, 0))
        image = np.clip(image, 0, 1) * 255
        image = image.astype('uint8')
        mask = mask_generator.generate(image)

        del sam, mask_generator
        torch.cuda.empty_cache()

        sorted_anns = sorted(mask, key=(lambda x: x['area']), reverse=True)
        img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))

        count = 1
        for ann in sorted_anns:
            m = ann['segmentation']
            img[m] = count
            count += 1

        self.model.to(device="cuda")
        self.model.eval()

        input = torch.tensor(input).to(device).unsqueeze(0)
        input.requires_grad_(True)
        output = self.model(input)

        score, indices = torch.max(output, 1)
        grad_output = torch.autograd.grad(score, input, create_graph=True)[0]
        # grad_output = self.ig.attribute(input, target=indices)

        np_grad_output = grad_output.detach().cpu().numpy()
        flattened_arr = np_grad_output.flatten()
        sorted_indices = np.argsort(flattened_arr)[::-1]

        count = 0
        sets = {}
        for flat_index in sorted_indices:
            indices = np.unravel_index(flat_index, grad_output.shape)

            if img[indices[2]][indices[3]] not in sets:
                # temp_attr = self.make_attribution(input, indices, self.max_features, grad_output[indices].item()*threshold)
                temp_attr = self.make_attribution(input, indices, self.max_features, threshold)
                if np.count_nonzero(temp_attr) != 0:
                    sets[img[indices[2]][indices[3]]] = temp_attr
                    # print("[INFO] Set " + str(count) + " Created with " + str(np.count_nonzero(sets[img[indices[2]][indices[3]]])))
                    count+=1

            if count == k:
                break

        return sets

    def attribute_set_ig(self, input, mask, steps=30):
        '''
            Find importance score from a given mask (interaction set)
            using Integrated Directional Gradients
        '''
        np_image = np.transpose(input.cpu().detach().numpy(), (1, 2, 0))
        img = np.zeros(np_image.shape)
        mag = np.zeros(np_image.shape)
        m = np.transpose(mask, (1, 2, 0))
        img = m
        img *= np_image
        mag = np.linalg.norm(img)
        img = img/mag
        baseline = 0*input
        
        scaled_inputs = [baseline + (float(i)/steps)*(input-baseline) for i in range(0, steps+1)]
        grads = []
        for inp in scaled_inputs:
            inp = inp.detach()
            grads.append(img*np.transpose(attributions.saliency(self.model, inp.unsqueeze(0)), (1, 2, 0)))
        idg = np.average(grads, axis=0)
        return np.linalg.norm(idg)

    def attribute(self, inputs, threshold=0.50, steps=30, k=5, max_samples=30):
        idgs = []
        for image in inputs:
            start_time = time.time()
            sets = self.find_feature_interaction_sets_sam(image, k, threshold)
            # print("[INFO] Sets Found")
            final_attr = []
            for mask in sets.values():
                value_set = 0
                for i in range(max_samples):
                    sample_mask = mask*np.random.randint(0, 2, size=mask.shape)
                    value_set += self.attribute_set_ig(image, sample_mask)
                # print("[INFO] Importance Score of Set with", np.count_nonzero(mask), "features:", value_set)
                attr_set = mask*(value_set)
                # print("[INFO] Attribution Set", attr_set)
                final_attr.append(attr_set)
            # print("[INFO] Final Attribution Set", final_attr)
            idg = np.nansum(final_attr, axis=0)
            # print("[INFO] Sets Attributed")
            idgs.append(idg)
            print(time.time() - start_time)
        idgs = np.array(idgs, dtype=object)
        return idgs
