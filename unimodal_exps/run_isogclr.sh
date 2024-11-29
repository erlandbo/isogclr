data_folder=~/Datasets/imagenet100/
data=imagenet100
rho=0.4
gamma=0.8
epoch=400
tau_init=0.7
beta_u=0.8

CUDA_VISIBLE_DEVICES=0 python main_supcon.py \
    --batch_size 256 \
    --learning_rate 1.2 \
    --cosine --warm \
    --epochs ${epoch} \
    --model resnet50_ \
    --dataset $data \
    --data_folder $data_folder \
    --size 224 \
    --desc dro_${data}_${rho}_${gamma}_${epoch}ep \
    --DRO_mod \
    --DRO_tau_init ${tau_init} \
    --DRO_gamma ${gamma}\
    --DRO_rho ${rho}\
    --DRO_beta_u ${beta_u} \
    --use_amp\
    --method SimCLR > isogclr_${data}_${rho}_${gamma}_${epoch}ep.log