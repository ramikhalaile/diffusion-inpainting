from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

image = Image.open('../outputs/chair.png')

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

questions = [
    "Question: What is the setting and environment in this image? Answer:",
    "Question: Describe the background, surfaces, lighting and atmosphere in this image. Answer:",
    "Question: Where is this scene taking place and what does the environment look like? Answer:",
    "Question: What type of location is this and what are its visual characteristics? Answer:",
    "Question: Describe the scene context including the floor, walls, lighting and surroundings. Answer:",
]

for q in questions:
    inputs = processor(images=image, text=q, return_tensors="pt").to("cuda", torch.float16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    answer = processor.decode(output[0], skip_special_tokens=True).strip()
    print(f"Q: {q}")
    print(f"A: {answer}")
    print()