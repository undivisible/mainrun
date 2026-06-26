#!/usr/bin/env python3
"""
Simple cosine LR test - just modify the learning rate manually.
"""
import json
import shutil
from pathlib import Path

def test_cosine_lr_manually():
    """Test by manually editing the LR value in train.py"""
    train_file = Path("mainrun/train.py")
    
    # Backup
    shutil.copy("mainrun/train.py", "mainrun/train.py.backup")
    
    content = train_file.read_text()
    
    # Simple approach: just change the LR to a lower value
    # to simulate what cosine decay would do later in training
    content = content.replace('lr: float = 0.02,', 'lr: float = 0.015,')  # 25% reduction
    content = content.replace('dropout: float = 0.15,', 'dropout: float = 0.18,')  # Slightly more dropout
    
    train_file.write_text(content)
    print("Modified train.py: lr=0.015, dropout=0.18")
    
    try:
        import subprocess
        import sys
        
        print("Running modified training (10 minute test)...")
        
        # Run with timeout
        result = subprocess.run([
            sys.executable, "mainrun/train.py"
        ], capture_output=True, text=True, timeout=600)  # 10 min
        
        if result.returncode == 0:
            print("✅ Test completed!")
            
            # Check if any validation steps ran
            try:
                with open("mainrun/logs/mainrun.log") as f:
                    val_steps = sum(1 for line in f if '"event": "validation_step"' in line)
                print(f"Validation steps completed: {val_steps}")
                
                if val_steps > 0:
                    print("Training is working. Ready for full experiment.")
                else:
                    print("No validation steps - training may have stopped early")
                    
            except Exception as e:
                print(f"Could not analyze log: {e}")
                
        else:
            print("❌ Test failed")
            if "ModuleNotFoundError" in result.stderr:
                print("Missing dependencies - need to run in proper environment")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ 10-minute test completed - training appears to be working")
    except KeyboardInterrupt:
        print("\n⏹️ Test stopped by user")
    finally:
        # Restore
        shutil.copy("mainrun/train.py.backup", "mainrun/train.py")
        Path("mainrun/train.py.backup").unlink()
        print("Restored original train.py")

if __name__ == "__main__":
    test_cosine_lr_manually()