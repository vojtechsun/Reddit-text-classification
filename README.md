# AITA Reddit Post Classifier

A binary text classification model that predicts whether a Reddit post from [r/AITAH](https://www.reddit.com/r/AITAH/) will be judged **NTA** (Not the A-hole) or **YTA** (You're the A-hole) based on the post content.

The model fine-tunes [RoBERTa-base](https://huggingface.co/roberta-base) on ~250 000 posts scraped from the PushshiftIO archive, using weighted cross-entropy loss to handle the class imbalance between NTA and YTA verdicts.

## How It Works

Each post on r/AITAH describes an interpersonal conflict and the community votes on whether the author was in the wrong. The flair assigned to the post (`Not the A-hole` / `Asshole`) serves as the ground-truth label. This project trains a classifier to predict that verdict from the post's title and text alone.

## Project Structure

```
├── Reddit_classification.ipynb   # Full training pipeline
├── processed_sample_250000-1.csv # Training data (not included — too large for GitHub)
├── final_model_roberta-base/     # Saved model after training (not included — too large for GitHub)
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   ├── label_mappings.json
│   ├── training_config.json
│   └── test_predictions.csv
├── confusion_matrix.png          # Generated evaluation plot
├── prediction_confidence.png     # Generated confidence distribution
└── token_length_distribution.png # Generated token length histogram
```

## Pipeline Overview

The notebook is organized into sequential phases:

1. **Configuration** — Model selection (`roberta-base` or `albert-base-v2`), file paths, label mapping, and all hyperparameters in one place for easy tuning.
2. **Data preprocessing** — Loads the CSV, combines post title + body text, cleans whitespace, and truncates extremely long posts (>10 000 chars).
3. **Label filtering & encoding** — Keeps only the two decisive labels (`Not the A-hole` → 0, `Asshole` → 1), dropping ambiguous verdicts.
4. **Stratified split** — 75 / 10 / 15 train / validation / test split, stratified by label to preserve class ratios.
5. **Tokenization** — RoBERTa tokenizer with truncation at 512 tokens. Dynamic padding via `DataCollatorWithPadding`.
6. **Model & weighted loss** — RoBERTa-base with a classification head. A custom `WeightedTrainer` applies an 8× penalty on YTA misclassifications to counter the NTA-heavy class imbalance.
7. **Training** — 4 epochs with AdamW, linear warmup, and epoch-level evaluation. Best model selected by macro F1.
8. **Evaluation** — Classification report, confusion matrix, and prediction confidence analysis on the held-out test set.
9. **Export** — Saves the model, tokenizer, label mappings, and test predictions for downstream deployment.
10. **Inference function** — A ready-to-use `predict_text(title, text)` function for integration into applications.

## Requirements

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn transformers datasets
```

A GPU or TPU is strongly recommended. The notebook was developed on Google Colab with TPU runtime. If running on TPU/XLA, the notebook already sets `optim="adamw_torch"` to avoid fused-optimizer incompatibility.

## Data

Training data is sourced from the [PushshiftIO archive](https://pushshift.io/signup). The expected CSV format uses `;` as delimiter and must contain at least these columns:

| Column | Description |
|---|---|
| `title` | Post title |
| `selftext` | Post body text |
| `link_flair_text` | Community verdict (`Not the A-hole`, `Asshole`, etc.) |

Place your CSV in the project root and update `DATA_FILE` in the configuration cell.

## Configuration

All tunable parameters live in a single config cell at the top of the notebook:

```python
MODEL_NAME = "roberta-base"

TRAIN_CONFIG = {
    'test_size': 0.15,
    'val_size': 0.10,
    'max_length': 512,
    'batch_size_train': 32,
    'batch_size_eval': 64,
    'learning_rate': 1e-5,
    'num_epochs': 4,
    'warmup_steps': 500,
    'weight_decay': 0.05,
    'random_state': 42
}
```

Class weights for the weighted loss are set in the model setup cell (`[1.0, 8.0]` for NTA/YTA).

## Usage

### Training

Open `Reddit_classification.ipynb` and run all cells sequentially. The trained model will be saved to `./final_model_roberta-base/`.

### Inference

After training (or after loading a saved model):

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("./final_model_roberta-base")
tokenizer = AutoTokenizer.from_pretrained("./final_model_roberta-base")

# Use the predict_text function from the notebook
label, confidence = predict_text(
    title="AITA for refusing to share my food?",
    text="I bought lunch for myself and my coworker asked for some. I said no.",
    model=model,
    tokenizer_obj=tokenizer
)
print(f"{label} ({confidence:.1%})")
```

## Known Limitations

- **Sample size** — Only 250 000 posts were used due to hardware constraints; the full archive is significantly larger.
- **Truncation** — Posts longer than 512 tokens (~36% of the dataset) are truncated, potentially losing context.
- **Class imbalance** — NTA posts outnumber YTA posts. The weighted loss helps but the weights were tuned manually rather than through systematic search.
- **Hyperparameter tuning** — No automated hyperparameter optimization was performed; further gains are likely achievable.

## What's Not in This Repo

The trained model (`final_model_roberta-base/`) and the training dataset (`processed_sample_250000-1.csv`) exceed GitHub's file size limits and are not included in this repository. To reproduce the results, you will need to:

1. Obtain the data from the [PushshiftIO archive](https://pushshift.io/signup) and prepare a CSV in the format described in the Data section above.
2. Run the notebook end-to-end to train and save the model locally.

Make sure your `.gitignore` excludes these files:

```gitignore
# Large files
*.csv
final_model_*/
results/
logs/
```

## License

This project was created as part of a group project. The model and training pipeline are provided as-is were fully created by myself with assistance of artificial intelligence(claude) for visualization and some formating of the code.
