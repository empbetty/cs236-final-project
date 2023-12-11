import argparse
import os
import re
import torch
from diffusers import DiffusionPipeline

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
		default=30,
		help=("inference steps"),
	)

	args = parser.parse_args()
	return args

def generate_images(prompt, model_path, steps, output_folder):
	pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
	for i in range(5):
		image = pipeline(prompt, num_inference_steps=steps, guidance_scale=7.5, height=512, weight=512).images[0]
		file_name = re.sub(r'\W+', '-', prompt)
		file_name = output_folder + "/" + file_name + "-" + str(i) + ".png"
		os.makedirs(os.path.dirname(file_name), exist_ok=True)
		image.save(file_name)
		print("image saved to " + file_name)

def main():
	PROMPTS = [
	"oil painting of sks dog in style of van gogh",
	"sks dog ultra realistic portrait, high definition, 8k, vibrant color",
	"a cute sks dog in pastel crayon style",
	"a simple sketch of sks dog made in pencil, a minimum number of black lines on a white background",
	]

	args = parse_args()  # get arguments
	for prompt in PROMPTS:
		generate_images(prompt, args.model_path, args.steps, args.output_folder)

if __name__ == "__main__":
	main()

