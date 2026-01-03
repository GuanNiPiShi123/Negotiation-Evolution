# Negotiation-Evolution
Sft training example:
```bash
python3 sft_train.py  --dataset_repo ./nego_datasets/bargain/ --task bargain --output_dir output --lower_bound_metric outcome.gap_ratio --lower_bound -0.01 --per_device_train_batch_size 4 --per_device_eval_batch_size 2 --gradient_accumulation_steps 4 --num_train_epochs 4 --learning_rate 1e-5 --eval_steps 50 --logging_steps 100 --use_lora --use_4bit  --save_total_limit 4
```
MCTS Data Synthesis:
```bash
python3 generation_data.py --dataset_repo bargain --dataset_name bargain --evol_stage 1 --model_name ./checkpoint-592/  --max_new_turns 10  --num_mcts_sims 2 --max_realizations 3  --num_samples 2 --user_generation_kwargs '{"model": "gpt-4o-mini"}'  --assistant_generation_kwargs '{"model": "gpt-4o-mini"}' --reward_generation_kwargs '{"model": "gpt-4o-mini"}' --use_4bit --use_lora --add_system_prompt --evolution_ratio 0.3679 --use_mcts
```

GRPO Training:
```bash
python grpo_train.py --dataset_name bargain --metric_names "gap_ratio" "consistency" "bargain_interactivity" --metric_weights 0.4 0.3 0.3  --evol_stage 1 --user_generation_kwargs '{"model": "gpt-4o-mini"}' --assistant_generation_kwargs '{"model": "gpt-4o-mini", "temperature": 0.6}' --reward_generation_kwargs '{"model": "gpt-4o-mini"}' --dataset_repo ./nego_datasets/bargain/ --model_name outputs/sft/multiturn-bargain/Llama/checkpoint-epoch-2/ --output_dir outputs/grpo/multiturn-bargain/epoch1  --per_device_train_batch_size 1   --gradient_accumulation_steps 4  --num_train_epochs 1  --learning_rate 5e-6  --logging_steps 1  --wandb_entity yue-team   --wandb_project nego-evolution  --num_samples 3  --max_new_turns 13   --max_metric_workers 2  --use_4bit --use_lora
```
