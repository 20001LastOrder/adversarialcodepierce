# lang=python #programming language
# lr=5e-5
# batch_size=32
# beam_size=10
# source_length=256
# target_length=128
# data_dir=codegluex/dataset
# output_dir=results/code_summarization/$lang
# train_file=$data_dir/$lang/train.jsonl
# dev_file=$data_dir/$lang/valid.jsonl
# epochs=10 
# pretrained_model=microsoft/codebert-base #Roberta: roberta-base

# batch_size=64
# dev_file=$data_dir/$lang/valid.jsonl
# test_file=$data_dir/$lang/test.jsonl
# test_model=C:/Users/chenp/Documents/github/AdversaCodePierce/trained/code_summarization/checkpoint-best-bleu-python/pytorch_model.bin #checkpoint for test
# adv_candidate_file=results/randomness_codebert_variable_subtoken.json
# adv_mode=permutation

# echo $adv_mode

# python -m random_search_code_summarization.run  --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --output_dir $output_dir --max_target_length $target_length --beam_size $beam_size --adv_mode $adv_mode --adv_candidate_file $adv_candidate_file --adv_max_iterations 60 --test_filename $test_file --max_source_length $source_length --output_folder results/code_summarization/attack


# Code summarization
checkpoint=trained/code_summarization/t5_masked
base_model=Salesforce/codet5-base
test_filename=codegluex/dataset/python/test.jsonl
adv_candidate_file=results/randomness_t5_variables.json
max_source_length=256
max_target_length=128
output_folder=results/code_summarization/t5_masked

# random vector based attack
python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=random_vector

# permutation based attack
python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=permutation


python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=random

python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=heuristic

adv_candidate_file=results/randomness_t5_space.json

python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=random

python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=heuristic


# Code summarization
checkpoint=trained/code_summarization/t5
base_model=Salesforce/codet5-base
test_filename=codegluex/dataset/python/test.jsonl
adv_candidate_file=results/randomness_t5_variables.json
max_source_length=256
max_target_length=128
output_folder=results/code_summarization/t5

# random vector based attack
python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=random_vector

# permutation based attack
python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=permutation


python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=random

python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=heuristic

adv_candidate_file=results/randomness_t5_space.json

python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=random

python -m code_summarization_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_source_length $max_source_length --max_target_length $max_target_length --output_folder $output_folder --strategy=heuristic


# Bug detection
# checkpoint=trained/graphcodebert
# base_model=microsoft/graphcodebert-base
# test_filename=data/eval_synthetic/eval__VARIABLE_MISUSE__SStuB.txt-00000-of-00300
# adv_candidate_file=results/randomness_codebert_space.json
# max_length=512
# output_folder=results/bug_detection/graphcodebert

# echo "random attack on tokens"
# python -m bug_detection_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_length $max_length --output_folder $output_folder --strategy=random

# echo "heuristic attack on tokens"
# python -m bug_detection_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_length $max_length --output_folder $output_folder --strategy=heuristic

# echo "permutation attack"
# python -m bug_detection_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_length $max_length --output_folder $output_folder --strategy=permutation

# adv_candidate_file=results/randomness_codebert_variable.json

# echo "heuristic attack on variables"
# python -m bug_detection_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_length $max_length --output_folder $output_folder --strategy=heuristic

# echo "random attack on variables"
# python -m bug_detection_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_length $max_length --output_folder $output_folder --strategy=random


# echo "random vector attack on variables"
# python -m bug_detection_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_length $max_length --output_folder $output_folder --strategy=random_vector
