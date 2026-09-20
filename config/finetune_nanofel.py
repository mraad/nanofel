import time

out_dir = 'out-nanofel'
eval_interval = 200
eval_iters = 40
log_interval = 10
wandb_log = False
wandb_run_name = 'nanofel-' + str(time.time())

dataset = 'nanofel'
init_from = 'gpt2-medium'

always_save_checkpoint = False  # keep best val loss only

# 474k train tokens gold only, 1.37M with extra.jsonl; block 256 covers the longest pair (199 tokens) with room
block_size = 256
batch_size = 16
gradient_accumulation_steps = 2  # 2 GPUs -> 1 each, 1 GPU -> both; 32 seqs * 256 = 8k tok / iter

max_iters = 3000
lr_decay_iters = 3000
warmup_iters = 100
learning_rate = 3e-5
min_lr = 3e-6
decay_lr = True
dropout = 0.1
weight_decay = 0.1
