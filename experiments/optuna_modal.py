"""
Optuna Hyperparameter Search for PyReflect NR->SLD Model on Modal
Runs on T4 GPU with full dependency setup and W&B logging

Configured for Modal execution. Adjust parameters and paths as needed.
"""

import modal

# Configuration
N_TRIALS = 20
EPOCHS_PER_TRIAL = 7
WANDB_PROJECT = "pyreflect-overfitting-modal"

# Create Modal App
app = modal.App("pyreflect-optuna-search")

# Define container image with all dependencies AND data
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.1.0",
        "optuna",
        "wandb",
        "plotly",
        "pandas",
        "scikit-learn",
        "scipy",
        "opencv-python",
        "pyyaml",
        "tqdm",
        "refnx",
        "llvmlite",
        "numba"
    )
    .apt_install("git")
    .run_commands("pip install git+https://github.com/williamQyq/pyreflect.git")
    .add_local_dir(
        "../master_training_data",
        remote_path="/root/master_training_data",
        copy=True
    )
)


@app.function(
    gpu="T4",
    image=image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=20 * 60 * 60  # 20 hours in seconds
)
def run_optuna_search():
    """
    Main function that runs Optuna hyperparameter search on Modal
    """
    import optuna
    import wandb
    import numpy as np
    import torch
    import torch.nn as nn
    from dataclasses import dataclass
    from pathlib import Path
    from pyreflect.input import NRSLDDataProcessor, DataProcessor
    from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer
    import pyreflect.pipelines.reflectivity_pipeline as workflow
    
    print("🔍 Starting Optuna hyperparameter search on Modal...")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    # Data paths in Modal container
    DATA_ROOT = Path("/root/master_training_data")
    NR_FILE = str(DATA_ROOT / "data/curves/nr_train.npy")
    SLD_FILE = str(DATA_ROOT / "data/curves/sld_train.npy")
    NORM_STATS_FILE = str(DATA_ROOT / "data/normalization_stat.npy")
    
    print(f"Data root: {DATA_ROOT}")
    print(f"NR file: {NR_FILE}")
    print(f"SLD file: {SLD_FILE}")
    print(f"Normalization stats: {NORM_STATS_FILE}")
    
    @dataclass
    class SimpleTrainerParams:
        """Training parameters for a single trial"""
        layers: int
        dropout: float
        batch_size: int
        epochs: int
    
    def train_with_logging(X_train, y_train, trainer_params, wandb_run):
        """
        Training loop with W&B logging
        Returns train_losses and val_losses for all epochs
        """
        # Create trainer
        trainer = NRSLDModelTrainer(
            X=X_train,
            y=y_train,
            layers=trainer_params.layers,
            batch_size=trainer_params.batch_size,
            epochs=trainer_params.epochs,
            dropout=trainer_params.dropout
        )
        
        # Prepare data splits (80/20 train/validation)
        list_arrays = DataProcessor.split_arrays(X_train, y_train, size_split=0.8)
        tensor_arrays = DataProcessor.convert_tensors(list_arrays)
        train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = \
            DataProcessor.get_dataloaders(*tensor_arrays, batch_size=trainer_params.batch_size)
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(
            trainer.model.parameters(),
            lr=2.15481e-05,
            weight_decay=2.6324e-05
        )
        loss_fn = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        # Training loop
        for epoch in range(trainer_params.epochs):
            # Train and validate
            train_loss = trainer.train_model(trainer.model, train_loader, optimizer, loss_fn)
            val_loss = trainer.validate_model(trainer.model, valid_loader, loss_fn)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Calculate overfitting gap
            overfitting_gap = val_loss - train_loss
            
            # Log to W&B
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "overfitting_gap": overfitting_gap
            })
            
            print(f"  Epoch {epoch + 1}/{trainer_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, Gap: {overfitting_gap:.6f}")
        
        return train_losses, val_losses
    
    def objective(trial):
        """
        Optuna objective function - tries different hyperparameters
        Returns validation loss (lower is better)
        """
        # Suggest hyperparameters
        layers = trial.suggest_int("layers", 6, 12)
        dropout = trial.suggest_float("dropout", 0.3, 0.7)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number + 1}/{N_TRIALS}")
        print(f"Testing: layers={layers}, dropout={dropout:.2f}, batch_size={batch_size}")
        print(f"{'='*60}\n")
        
        # Initialize W&B for this trial
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"trial_{trial.number}_L{layers}_D{dropout:.2f}_B{batch_size}",
            config={
                "layers": layers,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": EPOCHS_PER_TRIAL,
                "trial_number": trial.number,
                "platform": "modal",
                "gpu": "T4"
            },
            reinit=True
        )
        
        try:
            # Load data
            print(f"Loading data from {NR_FILE}...")
            dproc = NRSLDDataProcessor(NR_FILE, SLD_FILE).load_data()
            
            # Preprocess data
            X_train, y_train = workflow.preprocess(dproc, NORM_STATS_FILE)
            print(f"Data loaded: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
            
            # Create trainer params
            trainer_params = SimpleTrainerParams(
                layers=layers,
                dropout=dropout,
                batch_size=batch_size,
                epochs=EPOCHS_PER_TRIAL
            )
            
            # Train model with logging
            train_losses, val_losses = train_with_logging(X_train, y_train, trainer_params, run)
            
            # Get final validation loss (what Optuna optimizes)
            final_val_loss = val_losses[-1]
            final_train_loss = train_losses[-1]
            overfitting_gap = final_val_loss - final_train_loss
            
            # Log final metrics to W&B
            wandb.log({
                "final_val_loss": final_val_loss,
                "final_train_loss": final_train_loss,
                "final_overfitting_gap": overfitting_gap
            })
            
            print(f"\n✅ Trial {trial.number} complete!")
            print(f"   Final validation loss: {final_val_loss:.6f}")
            print(f"   Final train loss: {final_train_loss:.6f}")
            print(f"   Overfitting gap: {overfitting_gap:.6f}")
            
            wandb.finish()
            
            # Return validation loss for Optuna to minimize
            return final_val_loss
            
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            wandb.finish()
            raise optuna.TrialPruned()
    
    # Main optimization routine
    print("\n" + "="*60)
    print("🔍 OPTUNA HYPERPARAMETER SEARCH ON MODAL")
    print("="*60)
    print(f"Project: {WANDB_PROJECT}")
    print(f"Trials: {N_TRIALS}")
    print(f"Epochs per trial: {EPOCHS_PER_TRIAL}")
    print(f"Search space:")
    print(f"  - Layers: 6-12")
    print(f"  - Dropout: 0.3-0.7")
    print(f"  - Batch size: [16, 32, 64]")
    print("="*60)
    
    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="pyreflect_overfitting_modal"
    )
    
    # Run optimization
    print("\n🚀 Starting optimization...")
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Check if any trials succeeded
    if len(study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in study.trials):
        print("\n❌ No trials completed successfully!")
        return {
            "success": False,
            "message": "No trials completed successfully"
        }
    
    # Results
    print("\n" + "="*60)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\n🏆 Best trial: #{study.best_trial.number}")
    print(f"   Validation loss: {study.best_value:.6f}")
    print(f"\n📊 Best parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Return results
    return {
        "success": True,
        "best_trial_number": study.best_trial.number,
        "best_val_loss": study.best_value,
        "best_params": study.best_params,
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }


@app.local_entrypoint()
def main():
    """
    Local entrypoint - runs the search remotely and prints results
    """
    print("🚀 Launching Optuna hyperparameter search on Modal T4 GPU...")
    print(f"Configuration: {N_TRIALS} trials, {EPOCHS_PER_TRIAL} epochs per trial")
    print(f"W&B Project: {WANDB_PROJECT}")
    print(f"Timeout: 20 hours\n")
    
    # Run the search remotely
    result = run_optuna_search.remote()
    
    # Print results
    print("\n" + "="*70)
    print("FINAL RESULTS FROM MODAL")
    print("="*70)
    
    if result["success"]:
        print(f"✅ Search completed successfully!")
        print(f"\n🏆 Best Trial: #{result['best_trial_number']}")
        print(f"   Validation Loss: {result['best_val_loss']:.6f}")
        print(f"\n📊 Best Parameters:")
        for key, value in result["best_params"].items():
            print(f"   {key}: {value}")
        print(f"\n📈 Trials completed: {result['n_trials_completed']}/{N_TRIALS}")
    else:
        print(f"❌ Search failed: {result['message']}")
    
    print("\n💡 Check W&B dashboard for detailed metrics and visualizations:")
    print(f"   https://wandb.ai/raheja-k-northeastern-university/{WANDB_PROJECT}")
    print("="*70)