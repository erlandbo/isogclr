data_folder=~/Datasets/imagenet100/
data=imagenet100
gamma=0.8
temp=0.07
epoch=400

CUDA_VISIBLE_DEVICES=0 python main_supcon.py \
    --batch_size 256 \
    --learning_rate 1.2 \
    --cosine --warm \
    --epochs ${epoch} \
    --model resnet50_ \
    --dataset $data \
    --data_folder $data_folder \
    --size 224 \
    --desc dro_sogclr_${data}_${temp}_${gamma}_${epoch}ep \
    --temp ${temp} \
    --BI_gamma ${gamma}\
    --BI_mod\
    --use_amp\
    --method SogCLR > sogclr_${data}_${temp}_${gamma}_${epoch}ep.log