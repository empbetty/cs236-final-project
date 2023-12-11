import argparse
import os
import re
import torch
from diffusers import StableDiffusionPipeline

def parse_args():
	parser = argparse.ArgumentParser(description="Generate Images of Dreambooth Finetuning")
	parser.add_argument(
		"--model_path",
		type=str,
		default="/home/empbetty/project/sddata/finetune/dreambooth/tangyuan",
		help=("the path to the trained model"),
	)

	parser.add_argument(
		"--output_folder",
		type=str,
		default="/home/empbetty/project/output/finetune/dreambooth/tangyuan",
		help=("the path to folder to hold generated images"),
	)

	parser.add_argument(
		"--steps",
		type=int,
		default=50,
		help=("inference steps"),
	)

	args = parser.parse_args()
	return args

def generate_images(prompt, model_path, steps, output_folder):
	pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
	pipeline.load_textual_inversion(model_path)
	for i in range(5):
		image = pipeline(prompt, num_inference_steps=steps, height=512, weight=512).images[0]
		file_name = re.sub(r'\W+', '-', prompt)
		file_name = output_folder + "/" + file_name + "-" + str(i) + ".png"
		os.makedirs(os.path.dirname(file_name), exist_ok=True)
		image.save(file_name)
		print("image saved to " + file_name)

def main():
	PROMPTS = ["oil painting of <tangyuan> dog style of van gogh"]

	args = parse_args()  # get arguments
	for prompt in PROMPTS:
		generate_images(prompt, args.model_path, args.steps, args.output_folder)

if __name__ == "__main__":
	main()

