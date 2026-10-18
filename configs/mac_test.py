out_dir = "out/mac_test"
dataset = "fineweb_edu"  # reads data/fineweb_edu/*.bin
device = "mps"
dtype = "float32"
compile = False
wandb_log = False

# tiny model
n_layer, n_head, n_embd = 2, 2, 128
block_size = 256
dropout = 0.0

# tiny run
batch_size = 8
gradient_accumulation_steps = 1
max_iters = 200
lr_decay_iters = 200  # cosine length = run length
warmup_iters = 20
learning_rate = 1e-3
eval_interval = 50
eval_iters = 20
log_interval = 10
