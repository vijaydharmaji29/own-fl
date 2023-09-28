#OFL File
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

torch.multiprocessing.set_sharing_strategy('file_system')


class Analyser:
    
    def __init__(self):
        print("\n\tInitialised Analyser Object\t\n")

    def calculate_dataset_mean(self, dataset_tensor):
        l = [data[0] for i, data in enumerate(dataset_tensor)]
        l = tuple(l)
        stacked_tensor = torch.stack(l)
        calculated_mean = torch.mean(stacked_tensor)
        return calculated_mean

    def calculate_dataset_std(self, dataset_tensor):
        l = [data[0] for i, data in enumerate(dataset_tensor)]
        l = tuple(l)
        stacked_tensor = torch.stack(l)
        calculated_std = torch.std(stacked_tensor)
        return calculated_std

    def cosine_similarity(self, tensor1, tensor2):
        cl = []

        cos = nn.CosineSimilarity(dim=0)

        for t in tensor1:
            cl.append(torch.norm(torch.flatten(cos(t, tensor2))))

        return cl

    def calculate_image_embeds(self, image_given):
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        image = image_given


        inputs = processor(text=["a photo of a man"], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds

        return(image_embeds)

    def dataset_embeds(self, dataset_pil):
        print("getting embeddings: ")
        dataset_embedings_list = []

        for i in range(len(dataset_pil)):
            image = dataset_pil[i][0]

            image_embed = self.calculate_image_embeds(image)
            if i == 0:
                print("Type of image embeds: ", type(image_embed))
                print("Image embeds: ", image_embed)

            dataset_embedings_list.append(image_embed)

        return dataset_embedings_list

    def dataset_mean_embeds(self, dataset_pil):
        dataset_embeds_list = self.dataset_embeds(dataset_pil)

        print("Done")

        return -1