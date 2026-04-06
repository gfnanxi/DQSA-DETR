checkpoint=$1
python main_aitod.py \
  --output_dir logs/DQSA_eval \
	-c config/DQSA_config.py --coco_path Datasets/aitod \
	--eval --resume $checkpoint \
	--options dn_scalar=100 embed_init_tgt=False \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

