$MODEL_NAME="runwayml/stable-diffusion-v1-5"
$INSTANCE_DIR="./datasets/cat_sneaker/instances/"
$CLASS_DIR="./datasets/cat_sneaker/class_dir/class_images/"
$CLASS_PROMPT_DIR="./datasets/cat_sneaker/class_dir/class_prompts"
$OUTPUT_DIR="./datasets/cat_sneaker/output/"

accelerate launch train_dreambooth.py `
  --pretrained_model_name_or_path=$MODEL_NAME  `
  --instance_data_dir=$INSTANCE_DIR `
  --class_data_dir=$CLASS_DIR `
  --class_prompt_dir=$CLASS_PROMPT_DIR `
  --output_dir=$OUTPUT_DIR `
  --with_prior_preservation --prior_loss_weight=1.0 `
  --train_text_encoder `
  --resolution=512 `
  --train_batch_size=2 `
  --mixed_precision="fp16"  `
  --gradient_accumulation_steps=2 --gradient_checkpointing `
  --learning_rate=1e-6 `
  --lr_scheduler="constant" `
  --lr_warmup_steps=200 `
  --num_class_images=200 `
  --max_train_steps=2000
