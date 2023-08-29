import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from transformers import (AutoModelForSemanticSegmentation,
                          SegformerImageProcessor)

processor = SegformerImageProcessor.from_pretrained(
    "mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained(
    "mattmdjaga/segformer_b2_clothes")

url = "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80"

image = Image.open(requests.get(url, stream=True).raw)
print(image.size)
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]

print(pred_seg.shape)

plt.imsave("output/test.png", pred_seg)

segments = torch.unique(pred_seg)  # Get a list of all the predicted items
for i in segments:
    mask = pred_seg == i  # Filter out anything that isn't the current item
    img = Image.fromarray((mask * 255).numpy().astype(np.uint8))
    name = model.config.id2label[i.item()]  # get the item name
    img.save(f"output/{name}.png", append=False)
