out_dir = "out/cluster_test"
dataset = "fineweb_edu"
device = "cuda"
dtype = "bfloat16"
compile = True
wandb_log = False
wandb_project = "llm_optimisation"

# ~11M non-embedding + ~19M tied embedding params
n_layer, n_head, n_embd = 6, 6, 384
block_size = 1024
dropout = 0.0

# 32 x 1024 x 4 = ~131k tokens per step
batch_size = 32
gradient_accumulation_steps = 4
max_iters = 4000
lr_decay_iters = 4000
warmup_iters = 100
learning_rate = 1e-3
min_lr = 1e-4  # 10x decay, as in Chinchilla
eval_interval = 250
eval_iters = 100
