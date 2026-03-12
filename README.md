# Transformer Fine-Tuning 

This repo contains code to **fine-tune a Transformer model (DistilBERT)** for multiple use cases.

---

# Project Structure

```
project-root/
│
├── notebooks/
│   └── text_classification_imdb.ipynb
│
├── models/             # fine-tuned models will be exported here
│   └── imdb_sentiment/ # imdb fine-tuned model for sentiment-analysis
│        ├── checkpoint-1563/
│        └── checkpoint-3126/
│
├── src/                # (optional future scripts)
│   └── train.py
│
├── data/               # optional dataset exports
│
├── pyproject.toml      # managed by uv
└── README.md
```

### Folder Explanation

**notebooks/**
Contains the Jupyter notebook used to:

* load the dataset
* tokenize text
* configure training
* fine-tune the model
* evaluate accuracy

**models/**
Stores model checkpoints created during training.

Example:

```
checkpoint-1563
checkpoint-3126
```

Each checkpoint contains:

```
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
trainer_state.json
training_args.bin
```

These checkpoints can be loaded directly for inference.

**src/** *(optional)*
Future location for training scripts outside notebooks.

**data/** *(optional)*
Local storage for exported datasets.

---

# Environment Setup

This project uses **uv** for dependency management.

## Install uv

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```
uv --version
```

---

## Install dependencies

From the project root:

```
uv add torch transformers datasets accelerate jupyter scikit-learn evaluate
```

**Note**: By default uv add torch installs CUDA 120. To install CUDA 130 (if supported), add the following in the pyproject.toml file

```
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
torchaudio = { index = "pytorch-cu130" }
```

Run this to check if correct CUDA version is being used
```
uv run python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

Main libraries used:

| Library      | Purpose                          |
| ------------ | -------------------------------- |
| transformers | Transformer models and pipelines |
| datasets     | Loading and processing datasets  |
| torch        | Deep learning backend            |
| accelerate   | GPU / training optimization      |
| jupyter      | Interactive notebooks            |
| evaluate     | evaluation metrics               |

---

# Running the Notebook

Start Jupyter from the project root:

```
uv run jupyter notebook
```

Open:

```
notebooks/text_classification_imdb.ipynb
```

The notebook performs:

1. Load IMDB dataset
2. Tokenize movie reviews
3. Load DistilBERT
4. Configure Trainer
5. Train for **2 epochs**
6. Save checkpoints

---

# Training Details

Model:

```
distilbert-base-uncased
```

Dataset:

```
IMDB movie reviews
```

Training configuration:

```
num_train_epochs = 2
evaluation_strategy = "epoch"
save_strategy = "epoch"
```

Training results:

| Epoch | Training Loss | Validation Loss | Accuracy |
| ----- | ------------- | --------------- | -------- |
| 1     | 0.2178        | 0.1950          | 92.6%    |
| 2     | 0.1339        | 0.2478          | 93.1%    |

Final validation accuracy: **~93%**

---

# Using the Trained Model

Because the Trainer saved checkpoints, load the latest checkpoint:

```
models/imdb_sentiment/checkpoint-3126
```

Example inference:

```python
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="../models/imdb_sentiment/checkpoint-3126"
)

classifier("This movie was fantastic!")
```

Example output:

```
[{'label': 'POSITIVE', 'score': 0.99}]
```

---

# Using GPU (Optional)

If CUDA is available:

```python
pipeline(
    "sentiment-analysis",
    model="../models/imdb_sentiment/checkpoint-3126",
    device=0
)
```

---

# Hugging Face Cache Locations

Downloaded assets are stored locally:

Models:

```
~/.cache/huggingface/hub
```

Datasets:

```
~/.cache/huggingface/datasets
```

These are reused automatically on subsequent runs.

---

# Training Concepts

### Epoch

One full pass through the training dataset.

Example:

```
25,000 reviews → model sees all once → epoch 1
25,000 reviews again → epoch 2
```

### Step

One gradient update from a batch.

Your run:

```
1563 steps per epoch
3126 total steps
```

### Accuracy

Percentage of correct predictions on the validation dataset.

Your model achieved:

```
~93% accuracy
```

---

# Future Improvements

Possible extensions:

* Train for more epochs
* Experiment with different models:

  * BERT
  * RoBERTa
  * TinyLlama
* Hyperparameter tuning
* Upload model to Hugging Face Hub
* Convert notebook into reproducible training scripts
* Add experiment tracking (Weights & Biases)

---

# Full Workflow

```
IMDB dataset
      ↓
tokenization
      ↓
DistilBERT
      ↓
fine-tuning (Trainer)
      ↓
model checkpoints
      ↓
pipeline inference
```
