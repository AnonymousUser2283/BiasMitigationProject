'''
Script to generate images with DreamBooth
'''
############################################## IMPORTS ###############################################################################
import os
import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
import random

device = 'cuda' if torch.cuda.is_available else 'cpu'

SEED = 42

####################################### GENERATE IMAGES WITH DREAMBOOTH ##############################################################

TRIAL_NUMBER = 54                  # number corresponding to the DB model chosen. To see to which training config this corresponds, check
                                  # colab/DB/{DISEASE}/output0.txt
CONFIG_NUMBER = 0                 # number corresponding to the generation config used for that particular DB model. To see to which
                                  # generation config this correspond, check colab/DB/{DISEASE}/gen{TRIAL_NUMBER}/config{CONFIG_NUMBER}/output.txt
DISEASE = 'scabbia'
BLACK_OR_BROWN = 'brown'
DISEASE_UNIQUE_ID = 'blckskn6' if BLACK_OR_BROWN == 'black' else 'brwnskn6'
PROMPT = f"Brown {DISEASE_UNIQUE_ID} human skin."
NEGATIVE_PROMPT = "eye, white, violet, blue, pale skin, fair skin, light skin, white skin"
GUIDANCE_SCALE = 7.5
NUM_IMAGES = 40

os.makedirs(f"colab/DB/{DISEASE}/model{TRIAL_NUMBER}", exist_ok=True)
os.makedirs(f"colab/DB/{DISEASE}/model{TRIAL_NUMBER}/{BLACK_OR_BROWN}", exist_ok=True)
os.makedirs(f"colab/DB/{DISEASE}/model{TRIAL_NUMBER}/{BLACK_OR_BROWN}/gen{CONFIG_NUMBER}", exist_ok=True)

output_file_path = f"colab/DB/{DISEASE}/model{TRIAL_NUMBER}/{BLACK_OR_BROWN}/gen{CONFIG_NUMBER}/output.txt"

with open(output_file_path, "a") as file:
    file.write(f"TRAINING CONFIG RETRIEVABLE AT TRIAL NUMBER: {TRIAL_NUMBER}" + "\n")
    file.write(f"GENERATION CONFIG: " + "\n")
    file.write(f"prompt: {PROMPT}" + "\n")
    file.write(f"negative prompt: {NEGATIVE_PROMPT}" + "\n")
    file.write(f"guidance scale: {GUIDANCE_SCALE}" + "\n")
    file.write(f"number of images: {NUM_IMAGES}" + "\n")
    file.write("scheduler: DDPMScheduler" + "\n")
    file.write(f"seed: {SEED}" + "\n")

pipe = StableDiffusionPipeline.from_pretrained(f"/leonardo_scratch/large/userexternal/cbellatr/DB/{DISEASE}/{BLACK_OR_BROWN}/dreambooth-model{TRIAL_NUMBER}", safety_checker=None, torch_dtype=torch.float16).to(device)
pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
generator = torch.Generator(device=device).manual_seed(SEED)
random.seed(SEED)

image_number_start = 0

with torch.autocast(device):
    # GENERATE!
    for i in range(88):
          NUM_IMAGES = 40
          images = pipe(PROMPT, num_inference_steps=50, height=512, width=512, guidance_scale=GUIDANCE_SCALE,\
                    num_images_per_prompt=NUM_IMAGES, negative_prompt=NEGATIVE_PROMPT, \
                    generator=generator).images

          generator=torch.Generator(device=device).manual_seed(SEED+i)

          for image_number, image in enumerate(images):
              image.resize((256,256))
              bl_br = 'bl' if BLACK_OR_BROWN=='black' else 'br'
              image_n = image_number_start + image_number
              image.save(f"colab/DB/{DISEASE}/model{TRIAL_NUMBER}/{BLACK_OR_BROWN}/gen{CONFIG_NUMBER}/{image_n:05}_{TRIAL_NUMBER}_{bl_br}_{CONFIG_NUMBER}.png")

          image_number_start += i*40