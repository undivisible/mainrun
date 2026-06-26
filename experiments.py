#!/usr/bin/env python3
"""
Experiment runner for pushing beyond 0.9746 validation loss.
Run with: python experiments.py
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

EXPERIMENTS = {
    "v9_lr_schedule": {
        "description": "Cosine decay with warmup - current model uses constant LR",
        "changes": {
            "lr_schedule": "cosine_warmup",
            "warmup_steps": 100,
            "max_lr": 0.025,
            "min_lr": 0.001
        },
        "hypothesis": "Better late-stage convergence, less overfitting"
    },
    
    "v10_wider_model": {
        "description": "Test if wider model helps (d_model 768, same params)",
        "changes": {
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 6,  # Reduce layers to keep params similar
            "block_size": 256,
            "vocab_size": 24576,
            "dropout": 0.12
        },
        "hypothesis": "Wider representations capture more patterns"
    },
    
    "v11_deeper_model": {
        "description": "Test if deeper model helps (10 layers, narrower)",
        "changes": {
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 10,
            "block_size": 256,
            "vocab_size": 24576,
            "dropout": 0.12
        },
        "hypothesis": "More hierarchical feature extraction"
    },
    
    "v12_flash_attention": {
        "description": "Implement flash attention variant",
        "changes": {
            "attention_type": "flash",
            "d_model": 576,
            "n_heads": 9,
            "n_layers": 8,
            "dropout": 0.15
        },
        "hypothesis": "Better attention scaling, faster training"
    },
    
    "v13_data_augment": {
        "description": "Add token-level data augmentation",
        "changes": {
            "token_dropout": 0.05,
            "token_shuffle": True,
            "d_model": 576,
            "dropout": 0.15
        },
        "hypothesis": "Better generalization through noise"
    },
    
    "v14_adaptive_dropout": {
        "description": "Increase dropout as training progresses",
        "changes": {
            "dropout_schedule": "adaptive",
            "initial_dropout": 0.05,
            "final_dropout": 0.2,
            "dropout_ramp_steps": 400
        },
        "hypothesis": "Stronger regularization later in training"
    }
}

def run_experiment(name: str, config: dict):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"Description: {config['description']}")
    print(f"Hypothesis: {config['hypothesis']}")
    print(f"{'='*60}")
    
    # Create experiment branch
    branch = f"experiment-{name}"
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    
    # Backup current config
    Path("train.py.bak").write_text(Path("train.py").read_text())
    
    # Apply changes to train.py
    apply_changes(config["changes"])
    
    # Run training
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/experiment_{name}_{timestamp}.log"
    
    try:
        result = subprocess.run([
            "python", "train.py",
            "--experiment", name,
            "--log_file", log_file
        ], capture_output=True, text=True, timeout=3600*6)  # 6 hour timeout
        
        if result.returncode == 0:
            print(f"✅ Experiment {name} completed successfully")
            # Extract best validation loss
            best_loss = extract_best_loss(log_file)
            print(f"Best validation loss: {best_loss}")
            
            # Record result
            record_result(name, config, best_loss, "success")
        else:
            print(f"❌ Experiment {name} failed")
            print(f"Error: {result.stderr}")
            record_result(name, config, None, "failed")
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {name} timed out")
        record_result(name, config, None, "timeout")
    
    # Restore original train.py
    Path("train.py").write_text(Path("train.py.bak").read_text())
    Path("train.py.bak").unlink()
    
    # Return to main branch
    subprocess.run(["git", "checkout", "main"], check=True)

def apply_changes(changes: dict):
    """Apply configuration changes to train.py"""
    train_py = Path("train.py").read_text()
    
    # Simple regex-based replacements (could be made more robust)
    for key, value in changes.items():
        if key == "lr_schedule":
            # Add learning rate scheduler
            if "cosine_warmup" in str(value):
                scheduler_code = """
# Cosine decay with warmup
def get_lr(step: int, warmup_steps: int = 100, max_lr: float = 0.025, min_lr: float = 0.001) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
"""
                train_py = train_py.replace("import math", f"import math{scheduler_code}")
                train_py = train_py.replace('lr: float = 0.02', 'lr: float = field(default_factory=lambda: get_lr(global_step))')
        
        elif key in ["d_model", "n_heads", "n_layers", "block_size", "vocab_size", "dropout"]:
            # Update hyperparameters
            pattern = f"{key}: int = "
            if key == "dropout":
                pattern = f"{key}: float = "
            
            for line in train_py.split('\n'):
                if pattern in line:
                    old_value = line.split('=')[1].strip()
                    train_py = train_py.replace(line, line.replace(old_value, str(value)))
                    break
    
    Path("train.py").write_text(train_py)

def extract_best_loss(log_file: str) -> float:
    """Extract best validation loss from log file"""
    try:
        with open(log_file) as f:
            best_loss = float('inf')
            for line in f:
                if '"event": "validation_step"' in line:
                    data = json.loads(line)
                    best_loss = min(best_loss, data['loss'])
            return best_loss
    except:
        return float('inf')

def record_result(name: str, config: dict, loss: float, status: str):
    """Record experiment result"""
    result = {
        "name": name,
        "description": config["description"],
        "hypothesis": config["hypothesis"],
        "changes": config["changes"],
        "best_loss": loss,
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    
    results_file = Path("experiment_results.json")
    results = []
    if results_file.exists():
        results = json.loads(results_file.read_text())
    
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))

def main():
    """Main experiment runner"""
    if len(sys.argv) < 2:
        print("Available experiments:")
        for name, config in EXPERIMENTS.items():
            print(f"  {name}: {config['description']}")
        print("\nUsage: python experiments.py <experiment_name|all>")
        return
    
    experiment = sys.argv[1]
    
    if experiment == "all":
        print("Running all experiments sequentially...")
        for name in EXPERIMENTS:
            run_experiment(name, EXPERIMENTS[name])
    elif experiment in EXPERIMENTS:
        run_experiment(experiment, EXPERIMENTS[experiment])
    else:
        print(f"Unknown experiment: {experiment}")

if __name__ == "__main__":
    main()