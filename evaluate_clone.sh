checkpoint=trained/code_clone/best_checkpoint
base_model=microsoft/codebert-base
adv_candidate_file=results/randomness_codebert_variable.json
max_length=512
output_folder=results/code_clone/codebert

python -m code_clone_attack --checkpoint $checkpoint --base_model $base_model --adv_candidate_file $adv_candidate_file --max_length $max_length --output_folder $output_folder --strategy=heuristic --log_freq 20
