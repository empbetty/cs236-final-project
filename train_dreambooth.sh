export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="project/tangyuan-dataset-background-removed"
export CLASS_DIR="project/sddata/finetune/dreambooth/tangyuan-background-removed/class"
export OUTPUT_DIR="project/sddata/finetune/dreambooth/tangyuan-background-removed"
export HUB_MODEL_ID="tangyuan-dreambooth-8"

accelerate launch --mixed_precision="fp16"  diffusers/examples/dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=500 \
  --validation_prompt="a photo of a sks dog" \
  --num_validation_images=4 \
  --validation_steps=100 \
  --checkpointing_steps=500 \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb