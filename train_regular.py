
# The name of the model to pretrain. Choose from names in ``litgpt.config``. Mutually exclusive with
# ``model_config``. (type: Optional[str], default: null)
model_name: pythia-1.4b

# A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
# ``model_config``. (type: Optional[Config], default: null)
model_config:

# Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
# /teamspace/jobs/<job-name>/share. (type: <class 'Path'>, default: out/pretrain)
out_dir: out/pretrain/regular_pythia1.4b

# The precision to use for pretraining. Possible choices: "bf16-true", "bf16-mixed", "32-true". (type: Optional[str], default: null)
precision: bf16-true
# Optional path to a checkpoint directory to initialize the model from.
# Useful for continued pretraining. Mutually exclusive with ``resume``. (type: Optional[Path], default: null)
initial_checkpoint_dir:

# Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
# from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
# ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
# (type: Union[bool, Literal["auto"], Path], default: False)
resume: True

# Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
data: Wiki40b

# Training-related arguments. See ``litgpt.args.TrainArgs`` for details
train:

  # Number of optimizer steps between saving checkpoints (type: Optional[int], default: 1000)
  save_interval: 1000

  # Number of iterations between logging calls (type: int, default: 1)
  log_interval: 10

  # Number of samples between optimizer steps across data-parallel ranks (type: int, default: 512)
  global_batch_size: 256

  # Number of samples per data-parallel rank (type: int, default: 4)
  micro_batch_size: 4

  # Number of iterations with learning rate warmup active (type: int, default: 2000)
  lr_warmup_steps: 100

  # Number of epochs to train on (type: Optional[int], default: null)
  epochs:

  # Total number of tokens to train on (type: Optional[int], default: 3000000000000)
  max_tokens: 3000000000

  # Limits the number of optimizer steps to run. (type: Optional[int], default: null)
  max_steps:

  # Limits the length of samples. Off by default (type: Optional[int], default: null)
  max_seq_length:

  # Whether to tie the embedding weights with the language modeling head weights. (type: Optional[bool], default: False)
  tie_embeddings:

  #   (type: Optional[float], default: 1.0)
  max_norm: 1.0

  #   (type: float, default: 4e-05)
  # min_lr: 1e-4
  min_lr: 5e-5

# Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details
eval:

  # Number of optimizer steps between evaluation calls (type: int, default: 1000)
  interval: 1000

  # Number of tokens to generate (type: Optional[int], default: null)
  max_new_tokens:

  # Number of iterations (type: int, default: 100)
  max_iters: 100

  # Whether to evaluate on the validation set at the beginning of the training
  initial_validation: false

  # Whether to evaluate on the validation set at the end the training
  final_validation: false

# Optimizer-related arguments
optimizer:

  class_path: torch.optim.AdamW
  
  init_args:
    
    #   (type: float, default: 0.001)
    lr: 1e-3
    # lr: 6e-4
    
    #   (type: float, default: 0.01)
    weight_decay: 0.1
    
    #   (type: tuple, default: (0.9,0.999))
    betas:
      - 0.9
      - 0.95

# How many devices/GPUs to use. Uses all GPUs by default. (type: Union[int, str], default: auto)
devices: auto

# How many nodes to use. (type: int, default: 1)
num_nodes: 1

# Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
# module require this. (type: Optional[Path], default: null)
tokenizer_dir: checkpoints/EleutherAI/pythia-1.4b

# The name of the logger to send metrics to. (type: Literal['wandb', 'tensorboard', 'csv'], default: tensorboard)
logger_name: tensorboard

# The random seed to use for reproducibility. (type: int, default: 42)
seed: 42
