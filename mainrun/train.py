import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import structlog
import torch
from torch.nn import functional as F
from tqdm import tqdm

import utils
from data import BPETokenizer, get_batch, get_titles, iter_full_split, pretokenize_corpus, train_tokenizer
from model import GPT, GPTConfig

logger = None


@dataclass
class Hyperparameters:
    block_size: int = 128
    batch_size: int = 64
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
    run_tag: str = "v1_qknorm_dropout0.05"
    onecycle_pct_start: float = 0.1
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1000.0


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

    batches = len(train_ids) // (args.block_size * args.batch_size)
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
    )
    model = GPT(cfg).to(device)
    logger.log("model_info", parameters_count=sum(p.numel() for p in model.parameters() if p.requires_grad))

    opt = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), str(device))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr,
        total_steps=max_steps,
        pct_start=args.onecycle_pct_start,
        div_factor=args.onecycle_div_factor,
        final_div_factor=args.onecycle_final_div_factor,
    )

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
    opt.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        micro_in_group = 0
        for i in tqdm(range(1, batches + 1), desc=f"Epoch {epoch}/{args.epochs}"):
            xb, yb, ptr = get_batch(train_ids, ptr, args.block_size, args.batch_size, device)
            remaining = batches - i + 1
            acc_div = min(args.gradient_accumulation_steps, remaining)
            _, loss = model(xb, yb)
            (loss / acc_div).backward()
            running_loss += loss.item()
            micro_in_group += 1

            if micro_in_group == args.gradient_accumulation_steps or i == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                elapsed = time.time() - t0
                logger.log(
                    "training_step",
                    step=step,
                    max_steps=max_steps,
                    loss=running_loss / micro_in_group,
                    lr=opt.param_groups[0]["lr"],
                    elapsed_time=elapsed,
                    prnt=False,
                )
                running_loss = 0.0
                micro_in_group = 0
                if step == 1 or step % eval_interval == 0 or step == max_steps:
                    val_loss = evaluate()
                    logger.log(
                        "validation_step",
                        step=step,
                        max_steps=max_steps,
                        loss=val_loss,
                        elapsed_time=elapsed,
                    )


if __name__ == "__main__":
    try:
        main()
    finally:
        if logger and hasattr(logger, "file_handler"):
            logger.file_handler.close()