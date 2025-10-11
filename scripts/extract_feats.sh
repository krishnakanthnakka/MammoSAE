
python ./src/codebase/save_embeddings.py \
  --downstream_classifier_chk_path "/workspace/Mammo-CLIP/checkpoints/Downstream_evalualtion_b5_fold0/classification/Models/Models/Classifier/fine_tune/calcification/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_aucroc_ver084.pth" \
  --label "Suspicious_Calcification" \
  --sae_layer 39 \
  --split train \
  --num_images 4000

  