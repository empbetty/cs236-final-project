export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="project/tangyuan-dataset-background-removed"
export OUTPUT_DIR="project/sddata/finetune/textualInversion/tangyuan-background-removed"
export HUB_MODEL_ID="tangyuan-textual-inversion-2"

accelerate launch --mixed_precision="fp16" diffusers/examples/textual_inversion/textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<tangyuan>" \
  --initializer_token="dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-04 \
  --num_vectors=10 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="a photo of <tangyuan>" \
  --num_validation_images=4 \
  --validation_steps=100 \
  --checkpointing_steps=1000 \
  --push_to_hub \
  --hub_model_id=${HUB_MODEL_ID} \
  --report_to=wandb