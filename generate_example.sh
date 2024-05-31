lang=python #programming language
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=codegluex/dataset
output_dir=results/code_summarization/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=10 
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

batch_size=64
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=C:/Users/chenp/Documents/github/AdversaCodePierce/trained/code_summarization/checkpoint-best-bleu-python/pytorch_model.bin #checkpoint for test

adv_mode=random

echo $adv_mode

python -m random_search_code_summarization.run_example_generation  --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --output_dir $output_dir --max_target_length $target_length --beam_size $beam_size --adv_mode $adv_mode --adv_candidate_file results/randomness_codebert_variable_subtoken.json --adv_max_iterations 60 --test_filename $test_file --max_source_length $source_length --adv_idx 10