#!/usr/bin/env python3
"""
Training script with learning rate scheduling capabilities.
Based on train.py but adds cosine decay with warmup.
"""
import math
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

# Import existing modules
from model import GPT, GPTConfig
from data import load_data

@dataclass
class TrainConfig:
    """Training configuration with LR scheduling options"""
    # Model architecture (same as before)
    d_model: int = 576
    n_heads: int = 9
    n_layers: int = 8
    block_size: int = 256
    vocab_size: int = 24576
    dropout: float = 0.15
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 0.02
    adamw_lr: float = 0.0005
    max_steps: int = 896
    eval_interval: int = 28
    device: str = "mps"
    
    # LR scheduling
    lr_schedule: str = "constant"  # "constant", "cosine_warmup", "linear_decay"
    warmup_steps: int = 100
    min_lr: float = 0.001
    
    # Regularization
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    # EMA
    ema_decay: float = 0.999
    ema_update_freq: int = 10
    
    # Experiment tracking
    experiment_name: str = "baseline"
    log_file: str = "logs/mainrun.log"

def get_lr(step: int, config: TrainConfig) -> float:
    """Get learning rate for given step based on schedule"""
    if config.lr_schedule == "constant":
        return config.learning_rate
    
    elif config.lr_schedule == "cosine_warmup":
        if step < config.warmup_steps:
            return config.learning_rate * step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        return config.min_lr + (config.learning_rate - config.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    elif config.lr_schedule == "linear_decay":
        if step < config.warmup_steps:
            return config.learning_rate * step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        return config.learning_rate * (1 - progress) + config.min_lr * progress
    
    else:
        raise ValueError(f"Unknown LR schedule: {config.lr_schedule}")

class Trainer:
    """Enhanced trainer with LR scheduling"""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load data
        self.train_data, self.val_data = load_data()
        
        # Initialize model
        model_config = GPTConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            dropout=config.dropout
        )
        self.model = GPT(model_config).to(self.device)
        
        # Initialize optimizers
        self.muon = self._configure_muon()
        self.adamw = self._configure_adamw()
        
        # EMA model
        self.ema_model = self._create_ema_model()
        
        # Training state
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.log_file = open(config.log_file, 'a')
    
    def _configure_muon(self):
        """Configure Muon optimizer (simplified version)"""
        # For simplicity, using AdamW as placeholder
        # In practice, you'd import the actual Muon implementation
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.beta1, self.config.beta2)
        )
    
    def _configure_adamw(self):
        """Configure AdamW for embeddings and layernorms"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.adamw_lr,
            weight_decay=0,
            betas=(self.config.beta1, self.config.beta2)
        )
    
    def _create_ema_model(self):
        """Create EMA copy of model"""
        ema_model = GPT(self.model.config).to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        return ema_model
    
    def update_ema(self):
        """Update EMA model"""
        if self.step % self.config.ema_update_freq == 0:
            with torch.no_grad():
                for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                    ema_param.data.mul_(self.config.ema_decay).add_(param.data, alpha=1 - self.config.ema_decay)
    
    def get_batch(self, split: str):
        """Get training batch"""
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i+self.config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+self.config.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)
    
    @torch.no_grad()
    def estimate_loss(self):
        """Estimate loss on train and val sets"""
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.config.eval_interval)
            for k in range(self.config.eval_interval):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out
    
    def train_step(self):
        """Single training step with LR scheduling"""
        # Get current learning rate
        current_lr = get_lr(self.step, self.config)
        
        # Update optimizer learning rates
        for param_group in self.muon.param_groups:
            param_group['lr'] = current_lr
        for param_group in self.adamw.param_groups:
            param_group['lr'] = self.config.adamw_lr
        
        # Get batch and forward pass
        xb, yb = self.get_batch('train')
        
        # Forward pass
        logits, loss = self.model(xb, yb)
        
        # Backward pass
        self.muon.zero_grad(set_to_none=True)
        self.adamw.zero_grad(set_to_none=True)
        loss.backward()
        
        # Optimizer steps
        self.muon.step()
        self.adamw.step()
        
        # Update EMA
        self.update_ema()
        
        # Log
        elapsed = time.time() - self.start_time
        log_entry = {
            "event": "training_step",
            "timestamp": time.time(),
            "step": self.step,
            "max_steps": self.config.max_steps,
            "loss": loss.item(),
            "lr": current_lr,
            "adamw_lr": self.config.adamw_lr,
            "elapsed_time": elapsed,
            "prnt": self.step % 10 == 0
        }
        self.log_file.write(json.dumps(log_entry) + '\n')
        self.log_file.flush()
        
        if self.step % 10 == 0:
            print(f"step {self.step:4d}/{self.config.max_steps} | loss {loss.item():.4f} | lr {current_lr:.5f}")
        
        self.step += 1
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate model and check for new best"""
        losses = self.estimate_loss()
        val_loss = losses['val']
        
        # Evaluate with EMA model
        self.ema_model.eval()
        ema_losses = {}
        for split in ['train', 'val']:
            batch_losses = torch.zeros(self.config.eval_interval)
            for k in range(self.config.eval_interval):
                X, Y = self.get_batch(split)
                logits, loss = self.ema_model(X, Y)
                batch_losses[k] = loss.item()
            ema_losses[split] = batch_losses.mean()
        
        ema_val_loss = ema_losses['val'].item()
        
        # Log validation
        log_entry = {
            "event": "validation_step",
            "timestamp": time.time(),
            "step": self.step - 1,
            "max_steps": self.config.max_steps,
            "loss": ema_val_loss,
            "ema": True,
            "elapsed_time": time.time() - self.start_time
        }
        self.log_file.write(json.dumps(log_entry) + '\n')
        self.log_file.flush()
        
        print(f"step {self.step-1:4d}: val loss {ema_val_loss:.4f} (train {losses['train'].item():.4f})")
        
        # Check for new best
        if ema_val_loss < self.best_val_loss:
            self.best_val_loss = ema_val_loss
            print(f"New best validation loss: {self.best_val_loss:.6f}")
            
            # Save checkpoint
            checkpoint = {
                'model': self.ema_model.state_dict(),
                'config': self.config,
                'step': self.step,
                'val_loss': ema_val_loss,
                'best_val_loss': self.best_val_loss
            }
            torch.save(checkpoint, f'checkpoints/{self.config.experiment_name}_best.pt')
        
        return ema_val_loss
    
    def train(self):
        """Main training loop"""
        print(f"Starting training: {self.config.experiment_name}")
        print(f"LR schedule: {self.config.lr_schedule}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        
        self.start_time = time.time()
        
        for step in range(self.config.max_steps):
            self.train_step()
            
            if step % self.config.eval_interval == 0:
                self.evaluate()
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        self.log_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default='lr_test', help='Experiment name')
    parser.add_argument('--lr_schedule', default='cosine_warmup', help='LR schedule')
    parser.add_argument('--log_file', default='logs/lr_experiment.log', help='Log file')
    args = parser.parse_args()
    
    config = TrainConfig(
        experiment_name=args.experiment,
        lr_schedule=args.lr_schedule,
        log_file=args.log_file
    )
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()