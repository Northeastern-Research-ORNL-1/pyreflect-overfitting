# Optuna Hyperparameter Search Guide

## What This Does
Automatically finds the best CNN architecture (layers, dropout, batch size) to minimize overfitting in your NR→SLD model. Runs multiple trials and tracks everything in Weights & Biases.

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
        └── normalization_stat.npy
```

**Note:** The `optuna_search.py` script is in `experiments/` folder but looks for data in `../master_training_data/` (one level up).

---

## How to Run

### Start the Search
```bash
conda activate pyreflect
cd experiments
python optuna_search.py
```

### What Happens
- Runs N trials (configurable, default 1 for testing)
- Each trial tests different hyperparameters:
  - Layers: 6-12
  - Dropout: 0.3-0.7  
  - Batch size: 16, 32, or 64
- Trains each for specified epochs
- Logs everything to W&B dashboard

### Monitor Progress
- **Terminal:** See live trial updates with train/val loss
- **Browser:** https://wandb.ai → View training curves in real-time
- Each trial shows: `Trial X/N - Testing: layers=Y, dropout=Z, batch_size=W`

---

## Configuration

### Quick Test (Verify Setup)
For testing if everything works:
```python
N_TRIALS = 1          # Just 1 trial
EPOCHS_PER_TRIAL = 1  # Just 1 epoch
```
**Time:** ~10-15 minutes on CPU

### Full Search (GPU Recommended)
For actual hyperparameter optimization:
```python
N_TRIALS = 20         # Test 20 configurations
EPOCHS_PER_TRIAL = 7  # Train each for 7 epochs
```
**Time:** 
- On CPU: 5-8 hours (run overnight)
- On GPU: 40-100 minutes

### Thorough Search (If Time Permits)
```python
N_TRIALS = 50
EPOCHS_PER_TRIAL = 10
```
**Time:** Much longer, but finds better optimum

---

## Results

### Where to Find Them

1. **Terminal output** after completion:
```
   Best trial: #12
   Validation loss: 0.008955
   
   Best parameters:
     layers: 7
     dropout: 0.45
     batch_size: 32
```

2. **Text file:** `master_training_data/optuna_best_params.txt`

3. **Interactive plots:** `master_training_data/optuna_history.html` (only with 2+ trials)

4. **W&B dashboard:** Compare all trials side-by-side at https://wandb.ai

### Understanding Results

**Overfitting Gap = Validation Loss - Training Loss**

- **Negative (e.g., -0.0001):** Slight underfitting (good - model generalizes well)
- **~0 (e.g., 0.0002):** Perfect balance
- **Positive large (e.g., >0.01):** Overfitting (model memorizing training data)

**Goal:** Find configuration with lowest validation loss AND small overfitting gap

---

## Resume if Interrupted

If you stop the script (Ctrl+C), just run it again:
```bash
python optuna_search.py
```

Optuna automatically resumes from where it left off (saved in `optuna_study.db`).

You'll see: `Using an existing study with name 'pyreflect_overfitting'`

---

## Using the Best Parameters

After Optuna finishes, use the best parameters for your final model:

### Method 1: Update settings.yml
Edit `master_training_data/settings.yml`:
```yaml
nr_predict_sld:
  models:
    layers: 7        # From Optuna results
    dropout: 0.45    # From Optuna results  
    batch_size: 32   # From Optuna results
    epochs: 50       # Train longer for final model
```

Then train:
```bash
cd master_training_data
python -m pyreflect run --enable-sld-prediction
```

### Method 2: Use in Your Experiments
Copy the best parameters to your experiment folders:
```bash
# For exp2_reduced_layers/settings.yml
layers: 7
dropout: 0.45
batch_size: 32
epochs: 10
```

---

## Troubleshooting

### Error: "No module named optuna"
```bash
conda activate pyreflect
pip install optuna wandb plotly
```

### Error: "wandb not logged in"
```bash
wandb login
# Paste API key from https://wandb.ai/authorize
```

### Error: "File not found: data\curves\nr_train.npy"
**Issue:** Data files missing or script can't find them

**Solution:**
1. Check files exist:
```bash
   dir ..\master_training_data\data\curves
```
2. Verify you see `nr_train.npy` and `sld_train.npy`
3. If missing, generate data:
```bash
   cd ..\master_training_data
   python -m pyreflect run --enable-sld-prediction
```
   (Ctrl+C after data generation completes)

### Error: "Cannot evaluate parameter importances with only a single trial"
**Not an error!** Just a warning that appears when you run only 1 trial. Optuna needs 2+ trials to show which parameters matter most. This is normal for test runs.

### Trials taking too long?
- **Quick test:** Set `N_TRIALS=1`, `EPOCHS_PER_TRIAL=1`
- **Reduce scope:** Set `N_TRIALS=10`, `EPOCHS_PER_TRIAL=5`
- **Get GPU access:** 5-10x faster than CPU

### Script starts at "Trial 2/1" or "Trial 4/1"?
**This is normal!** Optuna remembers previous failed attempts. The study continues from where it left off. If you want to start fresh:
```bash
del optuna_study.db
python optuna_search.py
```

---

## What to Look For

### Good Signs ✅
- Validation loss decreases over trials
- Small overfitting gap (< 0.001)
- Best trial has val_loss < 0.01
- Multiple trials complete successfully

### Warning Signs ⚠️
- Validation loss stays high (> 0.02) across all trials
- Large overfitting gap (> 0.01) in best trial
- All trials fail (check data paths)
- Huge variation between trials (might need more epochs)

### On W&B Dashboard
Look for:
- Smooth training curves (not jumpy)
- Val loss following train loss closely
- Best trial clearly visible in comparison view

---

## File Structure After Running
```
pyreflect/
├── experiments/
│   ├── optuna_search.py
│   ├── optuna_study.db        ← Resume database
│   └── wandb/                 ← Local W&B logs
│       └── run-XXXXXX/
└── master_training_data/
    ├── optuna_best_params.txt  ← Best results
    ├── optuna_history.html     ← Optimization plot
    ├── optuna_importances.html ← Parameter importance
    └── data/
        └── curves/
```

---

## Tips for Best Results

1. **Start with quick test:** 1 trial, 1 epoch to verify setup
2. **Run full search on GPU:** Much faster (hours vs days)
3. **Use 20-50 trials:** Good balance of exploration and time
4. **Train 7-10 epochs per trial:** Enough to see overfitting trends
5. **Check W&B during run:** Catch issues early
6. **Don't delete optuna_study.db:** Allows resuming interrupted runs

---

## Advanced: Customizing Search Space

Edit `optuna_search.py` to change what Optuna optimizes:
```python
def objective(trial):
    # Current ranges
    layers = trial.suggest_int("layers", 6, 12)      # Try 6-12 layers
    dropout = trial.suggest_float("dropout", 0.3, 0.7)  # Try 0.3-0.7 dropout
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Add more parameters (optional)
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
```

---

**Last Updated:** February 2026