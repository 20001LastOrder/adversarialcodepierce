Bug detection
checkpoint=trained/graphcodebert
base_model=microsoft/graphcodebert-base
test_filename=data/eval_synthetic/eval__VARIABLE_MISUSE__SStuB.txt-00000-of-00300
adv_candidate_file=results/randomness_codebert_space.json
max_length=512
output_folder=results/bug_detection/graphcodebert

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


echo "random vector attack on variables"
python -m bug_detection_attack --checkpoint $checkpoint --base_model $base_model --test_filename $test_filename --adv_candidate_file $adv_candidate_file --max_length $max_length --output_folder $output_folder --strategy=random_vector
