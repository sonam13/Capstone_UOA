python /home/songhuatong/Reason-Searcher/train/reward_server_llama_stage1.py --data_path /opt/aps/workdir/sht-RAG_RL/datasets/stage_rl/stage1 --reward_pretrain /opt/aps/workdir/model/Meta-Llama-3.1-8B-Instruct --log_file /opt/aps/workdir/sht-RAG_RL/results/samples/llama_stage1.jsonl --port 1278


python /home/songhuatong/Reason-Searcher/train/reward_server_llama_stage2.py --data_path /opt/aps/workdir/sht-RAG_RL/datasets/stage_rl/stage2 --reward_pretrain /opt/aps/workdir/model/Meta-Llama-3.1-8B-Instruct --log_file /opt/aps/workdir/sht-RAG_RL/results/samples/llama_stage2.jsonl --port 1278


python /home/songhuatong/Reason-Searcher/train/reward_server_qwen_zero.py --data_path /opt/aps/workdir/sht-RAG_RL/datasets/stage_rl/stage2 --reward_pretrain /opt/aps/workdir/model/Qwen2.5-7B --log_file /opt/aps/workdir/sht-RAG_RL/results/samples/qwen.jsonl --port 1278
