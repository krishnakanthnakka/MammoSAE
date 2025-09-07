

if true; then
job_count=0
max_jobs=1
for label in Suspicious_Calcification; do
  for checkpoint_mode in pretrained finetuned; do
      python lib/intervention/sae_performance.py \
      --label $label \
      --sae_layer 39 \
      --split val \
      --split_for_top_concepts train \
      --num_images -1  \
      --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --sae_dataset vindr  --img_enc_name efficientb5 \
      --num_epochs 200 --hook_points layer_39 --resample_freq 10 --ckpt_freq 1 \
      --val_freq 1 --train_sae_bs 4096 --resample_dataset_size 20000 --checkpoint_mode $checkpoint_mode 
  done 
done
fi




if true; then
job_count=0
max_jobs=1
for label in Mass; do
  for checkpoint_mode in pretrained finetuned; do
      python lib/intervention/sae_performance.py \
      --label $label \
      --sae_layer 39 \
      --split val \
      --split_for_top_concepts train \
      --num_images -1  \
      --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --sae_dataset vindr  --img_enc_name efficientb5 \
      --num_epochs 200 --hook_points layer_39 --resample_freq 10 --ckpt_freq 1 \
      --val_freq 1 --train_sae_bs 4096 --resample_dataset_size 20000 --checkpoint_mode $checkpoint_mode 
  done 
done
fi
