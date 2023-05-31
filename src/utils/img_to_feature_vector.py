import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms as T

import warnings
warnings.filterwarnings("ignore")


class ImgToFeatureVector():
    RESNET_OUTPUT_SIZES = {
        'resnet18': 512,
        'resnet50': 2048,
    }

    def __init__(self, cuda=True, model='resnet-18', layer='default', layer_output_size=512, gpu=0):
        self.device = torch.device(f"cuda:{gpu}" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model

        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()


    def get_vec(self, img, tensor=True):
        transform = T.ToPILImage()
        img = transform(img)
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

        my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        with torch.no_grad():
            h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            return my_embedding.numpy()[0, :, 0, 0]
            
    
    def _get_model_and_layer(self, model_name, layer):
        model = getattr(models, model_name)(pretrained=True)
        if layer == 'default':
            layer = model._modules.get('avgpool')
            self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
        else:
            layer = model._modules.get(layer)
        return model, layer
     