"""
Optuna Hyperparameter Search for PyReflect NR->SLD Model
Finds optimal layers and dropout to reduce overfitting

Configured for local execution with W&B logging. Adjust paths and parameters as needed.
"""

import optuna
import wandb
import numpy as np
import torch
from pathlib import Path
from pyreflect.config import load_config
from pyreflect.input import NRSLDDataProcessor
from pyreflect.models.config import NRSLDModelTrainerParams
import pyreflect.pipelines.reflectivity_pipeline as workflow

print("🔍 Script started, loading libraries...")

# Configuration
PROJECT_ROOT = Path("../master_training_data")
print(f"✅ Libraries loaded, PROJECT_ROOT: {PROJECT_ROOT}")
WANDB_PROJECT = "pyreflect-overfitting"

#-----------------------------------------------------------------
N_TRIALS = 1  # Number of configurations to test
EPOCHS_PER_TRIAL = 1  # Reduced for faster trials
#-----------------------------------------------------------------

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
        name=f"trial_{trial.number}_L{layers}_D{dropout:.2f}",
        config={
            "layers": layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": EPOCHS_PER_TRIAL,
            "trial_number": trial.number
        },
        reinit=True
    )
    
    try:
        # Load data directly with absolute paths
        nr_file = str((PROJECT_ROOT / "data/curves/nr_train.npy").resolve())
        sld_file = str((PROJECT_ROOT / "data/curves/sld_train.npy").resolve())
        norm_stats_file = str((PROJECT_ROOT / "data/normalization_stat.npy").resolve())
        
        print(f"Loading data from: {nr_file}")
        dproc = NRSLDDataProcessor(nr_file, sld_file).load_data()
        
        # Preprocess data
        X_train, y_train = workflow.preprocess(dproc, norm_stats_file)
        
        # Create a simple params object with just what we need
        from dataclasses import dataclass
        
        @dataclass
        class SimpleTrainerParams:
            layers: int
            dropout: float
            batch_size: int
            epochs: int
        
        trainer_params = SimpleTrainerParams(
            layers=layers,
            dropout=dropout,
            batch_size=batch_size,
            epochs=EPOCHS_PER_TRIAL
        )
        
        # Train model (modified to return losses)
        train_losses, val_losses = train_with_logging(X_train, y_train, trainer_params, run)
        
        # Get final validation loss (what Optuna optimizes)
        final_val_loss = val_losses[-1]
        
        # Calculate overfitting metric (gap between train and val)
        overfitting_gap = val_losses[-1] - train_losses[-1]
        
        # Log final metrics to W&B
        wandb.log({
            "final_val_loss": final_val_loss,
            "final_train_loss": train_losses[-1],
            "overfitting_gap": overfitting_gap
        })
        
        print(f"\n✅ Trial {trial.number} complete!")
        print(f"   Final validation loss: {final_val_loss:.6f}")
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


def train_with_logging(X_train, y_train, trainer_params, wandb_run):
    """
    Modified training that logs to W&B during training
    Returns train_losses and val_losses for all epochs
    """
    from pyreflect.models.nr_sld_model_trainer import NRSLDModelTrainer
    from pyreflect.input import DataProcessor
    
    # Create trainer
    trainer = NRSLDModelTrainer(
        X=X_train,
        y=y_train,
        layers=trainer_params.layers,
        batch_size=trainer_params.batch_size,
        epochs=trainer_params.epochs,
        dropout=trainer_params.dropout
    )
    
    # Prepare data splits
    list_arrays = DataProcessor.split_arrays(X_train, y_train, size_split=0.8)
    tensor_arrays = DataProcessor.convert_tensors(list_arrays)
    train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader = \
        DataProcessor.get_dataloaders(*tensor_arrays, batch_size=trainer_params.batch_size)
    
    # Training loop with logging
    import torch.nn as nn
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=2.15481e-05, weight_decay=2.6324e-05)
    loss_fn = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(trainer_params.epochs):
        # Train
        train_loss = trainer.train_model(trainer.model, train_loader, optimizer, loss_fn)
        val_loss = trainer.validate_model(trainer.model, valid_loader, loss_fn)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Log to W&B
        wandb_run.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "overfitting_gap": val_loss - train_loss
        })
        
        print(f"Epoch {epoch + 1}/{trainer_params.epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    return train_losses, val_losses


def main():
    """Run Optuna hyperparameter search"""
    
    print("="*60)
    print("🔍 OPTUNA HYPERPARAMETER SEARCH")
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
        study_name="pyreflect_overfitting",
        storage="sqlite:///optuna_study.db",  # Saves progress to database
        load_if_exists=True  # Can resume if interrupted
    )
    
    # Run optimization
    print("\n🚀 Starting optimization...")
    print("This will take a while (probably hours?). You can safely interrupt (Ctrl+C) and resume later.\n")
    
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Check if any trials succeeded
    if len(study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in study.trials):
        print("\n❌ No trials completed successfully!")
        print("Check the errors above for details.")
        return
    
    # Print results
    print("\n" + "="*60)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"\n🏆 Best trial: #{study.best_trial.number}")
    print(f"   Validation loss: {study.best_value:.6f}")
    print(f"\n📊 Best parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Save results
    results_file = PROJECT_ROOT / "optuna_best_params.txt"
    with open(results_file, "w") as f:
        f.write(f"Best validation loss: {study.best_value:.6f}\n")
        f.write(f"Best parameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Generate optimization plots
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(str(PROJECT_ROOT / "optuna_history.html"))
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(str(PROJECT_ROOT / "optuna_importances.html"))
        
        print(f"📊 Visualizations saved to {PROJECT_ROOT}")
    except Exception as e:
        print(f"⚠️  Could not generate Optuna plots: {e}")


if __name__ == "__main__":
    main()