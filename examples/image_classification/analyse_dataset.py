#OFL File
import torch
import numpy as np
from torchvision import transforms

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def dataset_mean(dataset_tensor):
    l = [data[0] for i, data in enumerate(dataset_tensor)]
    l = tuple(l)
    stacked_tensor = torch.stack(l)
    # print("Dataset:", stacked_tensor)
    calculated_mean = torch.mean(stacked_tensor)
    return calculated_mean

def dataset_embeds(dataset_pil):
    print("Dataset PIL", dataset_pil[0])

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # image = Image.open(dataset_pil[0])
    image = dataset_pil[0][0]

    transform = transforms.Compose([transforms.PILToTensor()])
    # img_tensor = transform(image)


    inputs = processor(text=["a photo of a man", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    image_embeds = outputs.image_embeds
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    print("IMAGE")
    print(image_embeds)

    return -1
