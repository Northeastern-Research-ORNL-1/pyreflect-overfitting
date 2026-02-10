# Optuna Hyperparameter Search Guide

## What This Does
Automatically finds the best CNN architecture (layers, dropout, batch size) to minimize overfitting in your NR→SLD model. Runs 20 trials and tracks everything in Weights & Biases.

---

## Prerequisites

### 1. Install Dependencies
```bash
conda activate pyreflect
pip install optuna wandb plotly
wandb login  # Paste API key from https://wandb.ai/authorize
```

### 2. Prepare Data Folder
Ensure `master_training_data/` exists with:
```
master_training_data/
├── settings.yml
└── data/
    └── curves/
        ├── nr_train.npy
        └── sld_train.npy
```

---

## How to Run

### Start the Search
```bash
conda activate pyreflect
cd <project_root>
python experiments/optuna_search.py
```

### What Happens
- Runs 20 trials (5-8 hours on CPU)
- Each trial tests different hyperparameters:
  - Layers: 6-12
  - Dropout: 0.3-0.7  
  - Batch size: 16, 32, or 64
- Trains each for 7 epochs
- Logs everything to W&B dashboard

### Monitor Progress
- **Terminal:** See live trial updates
- **Browser:** https://wandb.ai → View training curves in real-time

---

## Results

### Where to Find Them

1. **Terminal output** after completion:
```
   Best parameters:
     layers: 9
     dropout: 0.45
     batch_size: 32
```

2. **Text file:** `master_training_data/optuna_best_params.txt`

3. **Interactive plots:** `master_training_data/optuna_history.html`

4. **W&B dashboard:** Compare all 20 trials side-by-side

---

## Customization

Edit these variables at the top of `optuna_search.py`:
```python
PROJECT_ROOT = Path("master_training_data")  # Your data folder
N_TRIALS = 20                                # Number of trials
EPOCHS_PER_TRIAL = 7                         # Epochs per trial
```

**For faster testing:**
- Reduce `N_TRIALS` to 10
- Reduce `EPOCHS_PER_TRIAL` to 5

**For thorough search:**
- Increase `N_TRIALS` to 50
- Increase `EPOCHS_PER_TRIAL` to 10

---

## Resume if Interrupted

If you stop the script (Ctrl+C), just run it again:
```bash
python experiments/optuna_search.py
```

Optuna automatically resumes from where it left off (saved in `optuna_study.db`).

---

## Using the Best Parameters

After Optuna finishes, use the best parameters for your final model:

### Update your `settings.yml`:
```yaml
nr_predict_sld:
  models:
    layers: 9        # From Optuna results
    dropout: 0.45    # From Optuna results  
    batch_size: 32   # From Optuna results
    epochs: 50       # Train longer for final model
```

### Train final model:
```bash
python -m pyreflect run --enable-sld-prediction --root master_training_data
```

---

## Troubleshooting

**Error: "No module named optuna"**
```bash
conda activate pyreflect
pip install optuna wandb plotly
```

**Error: "wandb not logged in"**
```bash
wandb login
# Paste API key from https://wandb.ai/authorize
```

**Error: "File not found: settings.yml"**
- Check `PROJECT_ROOT` path in script
- Ensure `master_training_data/settings.yml` exists

**Trials taking too long?**
- Reduce `EPOCHS_PER_TRIAL` to 5
- Reduce `N_TRIALS` to 10
- Run overnight

---

## What to Look For

### Good Signs ✅
- Validation loss decreases over trials
- Overfitting gap < 0.001 (val_loss - train_loss)
- Consistent performance across trials

### Warning Signs ⚠️
- Validation loss stays high (>0.02)
- Large overfitting gap (>0.01)
- High variance between trials

Check W&B dashboard to compare trial performance visually!

---

**Last Updated:** February 2026