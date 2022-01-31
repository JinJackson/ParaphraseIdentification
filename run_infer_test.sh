test_data='LCQMC'
model_type='baseline'
model_path='bert-base-uncased'
CUDA_VISIBLE_DEVICES=$1 python3 infer_test.py --test_file $test_data --model_type $model_type --model_path $model_path

#--test_data "LCQMC" --model_type "vae" --model_path "bert-base-chinese" --test_set "all"
