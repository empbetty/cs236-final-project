import argparse
import re
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Images from Lora Weights")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of a white dog on the grass",
        help="prompt to generate the image",
    )

    parser.add_argument(
        "--lora_model_path",
        type=str,
        default="/home/empbetty/project/sddata/finetune/lora/dogSimilarToTangyuan",
        help=("the path to the trained model file"),
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="/home/empbetty/project/output/finetune/lora/dogSimilarToTangyuan",
        help=("the path to folder to hold generated images"),
    )

    parser.add_argument(
        "--file_name",
        type=str,
        default="",
        help=("the file name of the generated image"),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help=("inference steps"),
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="scale of lora weights",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()  # get arguments
    if (args.file_name == ""):
        file_name = re.sub(r'\W+', '-', args.prompt)  # change all non-alphanumeric characters to dash
        file_name = file_name + '_' + str(args.scale) # append scale
        file_name = re.sub(r'\.', '_', file_name) # change . in scale to underscore
    else:
        file_name = args.file_name
    file_name = args.output_folder + "/" + file_name + ".png" # final file name

    model_base = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.unet.load_attn_procs(args.lora_model_path)
    pipe.to("cuda")
    
    image = pipe(args.prompt, num_inference_steps=args.steps, guidance_scale=7.5, cross_attention_kwargs={"scale": args.scale}).images[0]
    image.save(file_name)
    print("image saved to " + file_name)

if __name__ == "__main__":
    main()
