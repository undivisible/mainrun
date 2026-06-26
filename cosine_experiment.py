#!/usr/bin/env python3
"""
Run cosine LR experiment by modifying train.py in place.
"""
import json
import re
import shutil
from pathlib import Path

def modify_train_for_cosine_lr():
    """Modify train.py to use cosine LR schedule"""
    train_file = Path("mainrun/train.py")
    
    # Backup original
    shutil.copy("mainrun/train.py", "mainrun/train.py.backup")
    
    content = train_file.read_text()
    
    # Add cosine scheduler function after imports
    scheduler_func = '''
def cosine_lr_with_warmup(step: int, max_steps: int, warmup_steps: int = 100, 
                         max_lr: float = 0.025, min_lr: float = 0.001) -> float:
    """Cosine learning rate with warmup"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

'''
    
    # Insert after imports
    import_end = content.find("from data import load_data")
    if import_end != -1:
        insert_pos = content.find('\n', import_end) + 1
        content = content[:insert_pos] + scheduler_func + content[insert_pos:]
    
    # Replace learning rate field
    old_lr_field = 'lr: float = 0.02,'
    new_lr_field = 'lr: float = field(default_factory=lambda: cosine_lr_with_warmup(global_step, max_steps)),'
    content = content.replace(old_lr_field, new_lr_field)
    
    # Write back
    train_file.write_text(content)
    print("Modified train.py for cosine LR with warmup")

def restore_original_train():
    """Restore original train.py"""
    shutil.copy("mainrun/train.py.backup", "mainrun/train.py")
    Path("mainrun/train.py.backup").unlink()
    print("Restored original train.py")

def run_cosine_experiment():
    """Run cosine LR experiment"""
    print("Setting up cosine LR experiment...")
    
    # Modify train.py
    modify_train_for_cosine_lr()
    
    # Create experiment log
    log_file = "mainrun/logs/cosine_lr_experiment.log"
    
    print("Running cosine LR experiment...")
    print("This will take about 30 minutes. Use Ctrl+C to stop early.")
    
    try:
        import subprocess
        import sys
        
        # Run training with modified script
        result = subprocess.run([
            sys.executable, "mainrun/train.py"
        ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            print("✅ Cosine LR experiment completed!")
            
            # Extract best loss
            try:
                best_loss = float('inf')
                with open(log_file) as f:
                    for line in f:
                        if '"event": "validation_step"' in line:
                            data = json.loads(line)
                            best_loss = min(best_loss, data['loss'])
                print(f"Best validation loss: {best_loss}")
                
                # Compare with baseline
                baseline_loss = 0.974555
                improvement = ((baseline_loss - best_loss) / baseline_loss) * 100
                print(f"Improvement over baseline: {improvement:.2f}%")
                
                if best_loss < baseline_loss:
                    print("🎉 NEW BEST RESULT!")
                else:
                    print("No improvement over baseline")
                    
            except Exception as e:
                print(f"Could not extract best loss: {e}")
                
        else:
            print("❌ Experiment failed")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Experiment timed out")
    except KeyboardInterrupt:
        print("\n⏹️ Experiment stopped by user")
    finally:
        # Restore original
        restore_original_train()

if __name__ == "__main__":
    run_cosine_experiment()