# import model
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# model url: https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
# 470MB
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")


def model_pipeline(text: str, image: Image):
    # prepare inputs
    encoding = processor(image, text, return_tensors='pt')

    # check encoding
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()

    return model.config.id2label[idx]





