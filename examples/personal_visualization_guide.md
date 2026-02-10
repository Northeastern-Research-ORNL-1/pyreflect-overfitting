# Jupyter Notebook Visualization Guide for PyReflect Models (using example_reflectivity_pipeline)

## Prerequisites
- Trained model (`.pth` file)
- Normalization stats (`.npy` file)  
- Training or test data (`.npy` files)

---

## Steps to Follow

### 1. Launch Jupyter
```bash
conda activate pyreflect
cd <your_project_root>
jupyter notebook
```
Open `examples/example_reflectivity_pipeline.ipynb`

---

### 2. Section 1: Skip Installation Cells
- Already have pyreflect installed ✅

---

### 3. Section 2: Update Paths ⚠️ **MODIFY THIS**
```python
ROOT = Path("../experiments/YOUR_EXPERIMENT_NAME")  # Change experiment name
TRAIN_NR = ROOT / "data/curves/nr_train.npy"
TRAIN_SLD = ROOT / "data/curves/sld_train.npy"
PRETRAINED_MODEL = ROOT / "data/trained_nr_sld_model.pth"
NORMALIZATION_STAT = ROOT / "data/normalization_stat.npy"
TEST_NR = TRAIN_NR
TEST_SLD = TRAIN_SLD
```

---

### 4. Section 2.1: Update Config Root ⚠️ **MODIFY THIS**
```python
root = "../experiments/YOUR_EXPERIMENT_NAME"  # Change experiment name
config = pyreflect.config.load_config(root)
```

---

### 5. Section 3: Load Data
- Run as-is ✅

---

### 6. Section 4.1: Update Model Config ⚠️ **MODIFY THESE**
```python
model_config["layers"] = 12      # Your model's layers
model_config["dropout"] = 0.5    # Your model's dropout  
model_config["epochs"] = 10      # Epochs you trained
```

---

### 7. Section 4.2: Preprocess Data
- Run as-is ✅

---

### 8. Section 5: Load Model
```python
use_pretrained = True  # Keep this True
```
- Run as-is ✅

---

### 9. Section 6: Run Inference ⚠️ **OPTIONAL: Change test_batch**
```python
test_batch = 10  # Change to see more/fewer samples
```

---

### 10. Section 8: Compute NR from SLD
- Run as-is ✅

---

### 11. Section 9: Visualize
- Run as-is ✅
- See your plots! 🎨

---

## Summary: What to Change Each Time

| Section | What to Change | Example |
|---------|---------------|---------|
| 2 | `ROOT` path | `"../experiments/exp2_reduced_layers"` |
| 2.1 | `root` variable | Same as above |
| 4.1 | `layers`, `dropout`, `epochs` | Match your experiment config |
| 6 | `test_batch` (optional) | 10, 20, 50, etc. |

**Everything else: Run as-is** ✅

---

## Pro Tip: Save As New Notebook

For each experiment:
```
File → Save Notebook As...
Name: visualize_exp1.ipynb, visualize_exp2.ipynb, etc.
```

Then you can compare experiments side-by-side later!

---

## Expected Outputs

### Good Signs ✅
- NR curves (left plot): Original and computed lines overlap
- SLD profiles (right plot): Ground truth and prediction match closely

### Warning Signs ⚠️
- Large gaps between lines → Model needs more training or better architecture
- Predictions look very different from ground truth → Possible overfitting

---

## Troubleshooting

**Error: "File not found"**
- Check your ROOT path is correct
- Verify all 4 files exist in the experiment folder

**Error: "Kernel not found"**
- Make sure you selected "Python (pyreflect)" kernel

**Error: "Module not found"**
- Run: `conda activate pyreflect` and `pip install -e .` in project root

---

**Last Updated:** February 2026