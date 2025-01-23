# %% [markdown]
########################################## CONFIGURATION ###################################################################
DISEASE = "scabbia"
DISEASE_UNIQUE_ID = "brwnskn6"
BLACK_OR_BROWN = 'black' if DISEASE_UNIQUE_ID == "blckskn6" else 'brown'
INSTANCE_DATA_DIR = "colab/instance_dir_scabbia/brown"
CLASS_DATA_DIR = "colab/scabbia"
SEED = 42
TRIAL_NUMBER_INIT = 32

print(DISEASE, DISEASE_UNIQUE_ID, INSTANCE_DATA_DIR, CLASS_DATA_DIR, SEED, TRIAL_NUMBER_INIT)
########################################### LIBRARIES IMPORT ###############################################################
# %%
import os
import numpy as np
import cv2
from tqdm.notebook import tqdm
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.metrics import silhouette_score
from PIL import Image
from datasets import Dataset
from pathlib import Path
import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import random
from accelerate.utils import write_basic_config
write_basic_config()
import sys

# Add the directory containing the module to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'dreambooth'))

# Now you can import the module or function
from train_dreambooth import main, parse_args

# %%
########################################## GRID SEARCH INITIALIZATION #############################################################
import itertools

grid_params = {
     "learning_rate": [5e-07, 2e-06, 5e-06, 1e-05],
     "train_batch_size": [1],
     "instance_prompt": [f"{DISEASE_UNIQUE_ID}.", f"{DISEASE_UNIQUE_ID} human skin."],
     "max_train_steps": [2000, 3000, 4000, 5000],
     "train_text_encoder": [True],
     "class_data_dir": [False]
}

# Function to generate all possible combinations of the parameters
def generate_param_combinations(params):
    # Extract keys and values
    keys = params.keys()
    values = params.values()

    # Use itertools.product to compute all combinations
    combinations = list(itertools.product(*values))

    # Create a list of dictionaries from the combinations
    param_dicts = [dict(zip(keys, combination)) for combination in combinations]

    return param_dicts

# Generate all parameter combinations
param_combinations = generate_param_combinations(grid_params)

# Print the number of combinations and some examples
print(f"Total combinations: {len(param_combinations)}")

############################################### HYPERPARAMETER TUNING ############################################################
trial_number = TRIAL_NUMBER_INIT

output_file_path = f"colab/DB/{DISEASE}/output_{TRIAL_NUMBER_INIT}.txt"

with open(output_file_path, "a") as file:
    file.write("Performing grid search with the following hyperparameter values:" + "\n")
    for key, value in zip(grid_params.keys(), grid_params.values()):
        file.write(f"{key}: " + f"{value}" + "\n")

for param_combination in param_combinations:

    print("TRIAL NUMBER:", trial_number)

    custom_args = [
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        f"--instance_data_dir={INSTANCE_DATA_DIR}",
        f"--instance_prompt={param_combination['instance_prompt']}",
        f"--learning_rate={param_combination['learning_rate']}",
        f"--train_batch_size={param_combination['train_batch_size']}",
        f"--max_train_steps={param_combination['max_train_steps']}",
        "--lr_scheduler=constant",
        "--checkpointing_steps=100000",
        "--resolution=512",
        f"--seed={SEED}"
    ]

    if param_combination["train_text_encoder"]:
        custom_args.append("--train_text_encoder")

    if param_combination["class_data_dir"]:
        custom_args.append(f"--class_data_dir={CLASS_DATA_DIR}")
        custom_args.append("--with_prior_preservation")
        custom_args.append("--prior_loss_weight=1.0")
        custom_args.append(f"--class_prompt=Human skin.")

    # Replace sys.argv with the custom arguments
    sys.argv = ['dreambooth/train_dreambooth.py'] + custom_args

    # Parse the arguments as the script normally would
    args = parse_args(DISEASE, trial_number, BLACK_OR_BROWN)

    # Pass the parsed arguments to the main function
    main(DISEASE, trial_number, args)
    
    ################################################ GENERATING THE IMAGES #############################################################

    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # Create a StableDiffusionControlNetPipeline using the pre-trained models
    pipe = StableDiffusionPipeline.from_pretrained(
        f"/leonardo_scratch/large/userexternal/cbellatr/DB/{DISEASE}/{BLACK_OR_BROWN}/dreambooth-model{trial_number}",
        safety_checker = None,
        torch_dtype=torch.float16
    )
    pipe.to(device)

    # Use UniPCMultistepScheduler with the pipeline
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # %%
    # Set the random seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(SEED)
    # %%
    # Set the random seed for reproducibility
    random.seed(SEED)

    # Create a figure and axes for the grid
    fig, axes = plt.subplots(8, 5, figsize=(15, 24))
    fig.tight_layout(pad=3.0)

    prompt1 = f"Very black {DISEASE_UNIQUE_ID} human skin."
    prompt2 = f"Brown {DISEASE_UNIQUE_ID} human skin."
    negative_prompt = 'eye, white, violet, blue, pale skin, fair skin, light skin, white skin'
    guidance_scale = 7.5

    with torch.autocast(device):
        images = pipe([prompt1, prompt2], num_inference_steps=50, height=512, width=512, guidance_scale=guidance_scale,\
                      num_images_per_prompt=20, negative_prompt=[negative_prompt]*2, \
                        generator=generator).images

    for i in range(8):
        for j in range(5):
            # Generate an image
            with torch.autocast(device):
                image = images[5*i+j]
                image = image.resize((256, 256))

            # Display the image in the corresponding subplot
            axes[i, j].imshow(image)
            axes[i, j].axis("off")

    # Adjust layout and display the subplots
    plt.tight_layout()
    plt.savefig(f"colab/DB/{DISEASE}/all_samples{trial_number}.png")
    print(f"Samples saved at all_samples{trial_number}.png")

    ######################################################## SAVE CONFIGURATION TO FILE #########################################

    with open(output_file_path, "a") as file:
        file.write(f"##################################{trial_number}######################################" + "\n")
        for arg in custom_args:
            file.write(arg + "\n")
        file.write(f"Prompt1: {prompt1}" + "\n")
        file.write(f"Prompt2: {prompt2}" + "\n")
        file.write(f"Negative prompt: {negative_prompt}" + "\n")
        file.write(f"Guidance scale: {guidance_scale}" + "\n")
        file.write(f"Scheduler: DDPMScheduler" + "\n")

    print(f"Configuration saved to {output_file_path}")

    trial_number +=1