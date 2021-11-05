dataset="LCQMC"
model_type="hfl/chinese-bert-wwm"
batch_size=16
epochs=5
seed=2048
learning_rate='2e-5'
mlm=False
train_file="data/$dataset/tagging/train_tag.txt"
dev_file="data/$dataset/tagging/dev_tag.txt"
test_file="data/$dataset/tagging/test_tag.txt"
echo $train_file
CUDA_VISIBLE_DEVICES=6,7 python3 Train_VAE.py \
--train_file $train_file \
--dev_file $dev_file \
--test_file $test_file \
--save_dir "result/$dataset/$modeltype/cvae/bs$batch_size/epoch$epochs/seed$seed/$learning_rate/checkpoints" \
--model_type $model_type \
--do_train True \
--do_lower_case True \
--seed $seed \
--num_layers 2 \
--task_weight 0.9 \
--decoder_type "linear" \
--mask_rate 0.1 \
--mlm $mlm \
--learning_rate $learning_rate \
--epochs $epochs \
--batch_size $batch_size \
--max_length 128 \
--warmup_steps 0.1 \
