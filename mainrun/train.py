import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
from torch.nn import functional as F
from torch.optim._muon import Muon
from tqdm import tqdm

import utils
from data import BPETokenizer, get_batch, get_titles, iter_full_split, pretokenize_corpus, train_tokenizer
from model import GPT, GPTConfig

logger = None


@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 64
    train_batch_size: int = 32
    vocab_size: int = 16_000
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 384
    dropout: float = 0.05
    qk_norm: bool = True
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    evals_per_epoch: int = 3
    gradient_accumulation_steps: int = 1
    epochs: int = 7
    seed: int = 1337
    num_titles: int = 100_000
    val_frac: float = 0.10
    log_file: str = "./logs/mainrun.log"
    run_tag: str = "v5_bs32_dropout05_evalfix"
    muon_lr: float = 0.02
    adamw_lr: float = 5e-4
    muon_momentum: float = 0.95
    wsd_warmup_pct: float = 0.05
    wsd_decay_pct: float = 0.20
    rope_theta: float = 500.0
    label_smoothing: float = 0.0
    use_ema: bool = True
    ema_target_decay: float = 0.999


class ModelEMA:
    def __init__(self, model: torch.nn.Module, target_decay: float = 0.999):
        self.target = target_decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = None

    @staticmethod
    def _decay(step: int, target: float) -> float:
        return min(target, (1 + step) / (10 + step))

    def update(self, model: torch.nn.Module, step: int):
        d = self._decay(step, self.target)
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(d).add_(p.data, alpha=1 - d)

    def swap_in(self, model: torch.nn.Module):
        self.backup = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    def swap_out(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup = None


def make_wsd_lambda(max_steps: int, warmup_pct: float = 0.05, decay_pct: float = 0.20):
    warmup_steps = max(1, int(max_steps * warmup_pct))
    decay_start = max_steps - max(1, int(max_steps * decay_pct))
    decay_len = max(1, max_steps - decay_start)

    def lam(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        elif step < decay_start:
            return 1.0
        else:
            return max(0.0, 1.0 - (step - decay_start + 1) / decay_len)

    return lam


def configure_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = open(log_file, "w")
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    class DualLogger:
        def __init__(self, fh):
            self.file_handler = fh

        def log(self, event, **kwargs):
            log_entry = json.dumps({"event": event, "timestamp": time.time(), **kwargs})
            self.file_handler.write(log_entry + "\n")
            self.file_handler.flush()
            if kwargs.get("prnt", True):
                if "step" in kwargs and "max_steps" in kwargs:
                    tqdm.write(
                        f"[{kwargs.get('step'):>5}/{kwargs.get('max_steps')}] {event}: "
                        f"loss={kwargs.get('loss', 'N/A'):.6f} time={kwargs.get('elapsed_time', 0):.2f}s"
                    )
                else:
                    parts = [f"{k}={v}" for k, v in kwargs.items() if k not in ["prnt", "timestamp"]]
                    if parts:
                        tqdm.write(f"{event}: {', '.join(parts)}")
                    else:
                        tqdm.write(event)

    return DualLogger(file_handler)


def main():
    args = Hyperparameters()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.set_float32_matmul_precision("high")

    global logger
    logger = configure_logging(args.log_file)
    logger.log("hyperparameters_configured", **vars(args))
    logger.log("device_info", device=str(device))

    train_titles, val_titles = get_titles(args.num_titles, args.seed, args.val_frac)
    eos_token = "<eos>"
    tok = BPETokenizer(
        train_tokenizer(train_titles + val_titles, args.vocab_size, eos_token=eos_token),
        eos_token=eos_token,
    )
    train_ids, val_ids, _train_text, val_text = pretokenize_corpus(tok, train_titles, val_titles, eos_token)

    batches = len(train_ids) // (args.block_size * args.train_batch_size)
    opt_steps_per_epoch = math.ceil(batches / args.gradient_accumulation_steps)
    max_steps = args.epochs * opt_steps_per_epoch
    eval_interval = max(1, opt_steps_per_epoch // args.evals_per_epoch)
    logger.log(
        "dataset_info",
        titles_count=len(train_titles),
        epochs=args.epochs,
        batches_per_epoch=batches,
        opt_steps_per_epoch=opt_steps_per_epoch,
        tokens_per_epoch=len(train_ids),
        vocab_size=tok.vocab_size,
    )

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        eos_id=tok.eos_id,
        qk_norm=args.qk_norm,
        rope_theta=args.rope_theta,
        label_smoothing=args.label_smoothing,
    )
    model = GPT(cfg).to(device)
    logger.log("model_info", parameters_count=sum(p.numel() for p in model.parameters() if p.requires_grad))

    muon_params, adamw_params = model.get_optimizer_param_groups()
    muon_opt = Muon(
        muon_params,
        lr=args.muon_lr,
        weight_decay=args.weight_decay,
        momentum=args.muon_momentum,
    )
    adamw_opt = torch.optim.AdamW(
        adamw_params,
        lr=args.adamw_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    wsd_lam = make_wsd_lambda(max_steps, args.wsd_warmup_pct, args.wsd_decay_pct)
    muon_sched = torch.optim.lr_scheduler.LambdaLR(muon_opt, wsd_lam)
    adamw_sched = torch.optim.lr_scheduler.LambdaLR(adamw_opt, wsd_lam)
    logger.log(
        "optimizer_info",
        muon_params=len(muon_params),
        adamw_params=len(adamw_params),
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
    )

    ema = ModelEMA(model, target_decay=args.ema_target_decay) if args.use_ema else None

    def evaluate():
        model.eval()
        losses = 0.0
        with torch.no_grad():
            for xb, yb in iter_full_split(val_ids, args.block_size, args.batch_size, device):
                logits, _ = model(xb, yb)
                B, T, V = logits.size()
                loss = F.cross_entropy(logits.view(-1, V), yb.view(-1), reduction="sum")
                losses += loss.item()
        model.train()
        return losses / len(val_text)

    ptr = 0
    step = 0
    t0 = time.time()
    model.train()
    model.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        micro_in_group = 0
        for i in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.train_batch_size, device)
            remaining = batches - i + 1
            acc_div = min(args.gradient_accumulation_steps, remaining)
            _, loss = model(xb, yb)
            (loss / acc_div).backward()
            running_loss += loss.item()
            micro_in_group += 1

            if micro_in_group == args.gradient_accumulation_steps or i == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                muon_opt.step()
                adamw_opt.step()
                muon_sched.step()
                adamw_sched.step()
                model.zero_grad(set_to_none=True)
                step += 1
                if ema is not None:
                    ema.update(model, step)
                elapsed = time.time() - t0
                logger.log(
                    "training_step",
                    step=step,
                    max_steps=max_steps,
                    loss=running_loss / micro_in_group,
                    lr=muon_opt.param_groups[0]["lr"],
                    adamw_lr=adamw_opt.param_groups[0]["lr"],
                    elapsed_time=elapsed,
                    prnt=False,
                )
                running_loss = 0.0
                micro_in_group = 0
                if step == 1 or step % eval_interval == 0 or step == max_steps:
                    if ema is not None:
                        ema.swap_in(model)
                    val_loss = evaluate()
                    if ema is not None:
                        ema.swap_out(model)
                    logger.log(
                        "validation_step",
                        step=step,
                        max_steps=max_steps,
                        loss=val_loss,
                        ema=args.use_ema,
                        elapsed_time=elapsed,
                    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, "file_handler"):
            logger.file_handler.close()