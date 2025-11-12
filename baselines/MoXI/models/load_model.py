from transformers import ResNetForImageClassification, AutoImageProcessor
from transformers.models.vit.modeling_vit import ViTEmbeddings
from .mask_vit_embedding import MoXIViTForward, MoXIEmbeddingsForward, MoXIViTForImageClassificationForward
import os


# when using the mask method with a removal patch, vit_embedding is replaced.
def replace_vit_embedding_mask(args, model):
    if args.interaction_method == 'vit_embedding':
        print('vit embedding')
        model.vit.embeddings.forward = MoXIEmbeddingsForward.__get__(model.vit.embeddings, ViTEmbeddings)
        model.vit.forward = MoXIViTForward.__get__(model.vit, ViTModel)
        model.forward = MoXIViTForImageClassificationForward.__get__(model, ViTForImageClassification)
    elif args.interaction_method == 'pixel_zero_values':
        print('pixel zero input')
        pass
    return model

# load model checkpoint
def load_checkpoint(args):
    if args.model_name == 'vit-b':
        checkpoint = 'google/vit-base-patch16-224'
    elif args.model_name == 'vit-t':
        checkpoint = 'WinKawaks/vit-tiny-patch16-224'
    elif args.model_name == 'vit-s':
        checkpoint = 'WinKawaks/vit-small-patch16-224'
    elif args.model_name == 'deit-b':
        checkpoint = "facebook/deit-base-patch16-224"
    elif args.model_name == 'deit-t':
        checkpoint = "facebook/deit-tiny-patch16-224"
    elif args.model_name == 'deit-s':
        checkpoint = "facebook/deit-small-patch16-224"
    elif args.model_name == 'resnet':
        checkpoint = "microsoft/resnet-18"

    return checkpoint