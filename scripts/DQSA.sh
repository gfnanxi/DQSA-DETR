nohup python main_aitod.py \
  --output_dir logs/DQSA_detr -c config/DQSA_config.py \
  --pretrain_model_path dqsadetr_best305.pth\
  --finetune_ignore class_embed enc_out_class_embed \
  --options dn_scalar=100 embed_init_tgt=False \
  dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
  dn_box_noise_scale=1.0 > ddaqdetr_eval24.log  2>&1 &