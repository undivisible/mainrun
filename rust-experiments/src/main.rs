//! Main experiment runner

use clap::{Parser, Subcommand};
use rust_experiments::{get_device, experiments};
use tracing::{info, Level};

#[derive(Parser)]
#[command(name = "rust-experiments")]
#[command(about = "Rust-based GPT experiments matching Python suite", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all 6 experiments (v9-v14)
    All,

    /// v9: Cosine LR with warmup
    V9,
    /// v10: Wider model (d_model=768, 6 layers)
    V10,
    /// v11: Deeper model (d_model=512, 10 layers)
    V11,
    /// v12: Efficient attention with QK norm
    V12,
    /// v13: Token-level data augmentation
    V13,
    /// v14: Adaptive dropout scheduling
    V14,

    /// Run the WSD+EMA training (original experiment)
    Train {
        #[arg(long)]
        quick: bool,
        #[arg(long, default_value = "0.02")]
        max_lr: f32,
        #[arg(long, default_value = "945")]
        max_steps: usize,
        #[arg(long, default_value = "0.05")]
        warmup_pct: f32,
        #[arg(long, default_value = "0.20")]
        decay_pct: f32,
        #[arg(long)]
        no_ema: bool,
    },

    /// Run inference with a trained model
    Infer {
        #[arg(long, default_value = "checkpoints/model.safetensors")]
        checkpoint: String,
        #[arg(long, default_value = "data/tokenizer.json")]
        tokenizer: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "100")]
        max_tokens: usize,
        #[arg(long, default_value = "0.8")]
        temperature: f32,
        #[arg(long, default_value = "50")]
        top_k: usize,
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        #[arg(long)]
        speculative: bool,
        #[arg(long)]
        benchmark: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    let cli = Cli::parse();
    let device = get_device();

    info!("Rust Experiments Runner");
    info!("Device: {:?}", device);
    info!("");

    match cli.command {
        Commands::All => {
            info!("Running all experiments (v9-v14)...");
            let results = experiments::run_all(device);
            println!("\n{}", "=".repeat(60));
            println!("RESULTS SUMMARY");
            println!("{}", "=".repeat(60));
            for r in &results {
                println!("  {}: val={:.6} improvement={:.2}% status={}",
                    r.name, r.best_val_loss, r.improvement_percent, r.status);
            }
            let best = results.iter()
                .min_by(|a, b| a.best_val_loss.partial_cmp(&b.best_val_loss).unwrap());
            if let Some(b) = best {
                println!("\nBest: {} with val={:.6}", b.name, b.best_val_loss);
                if b.best_val_loss < 0.974555 {
                    println!("BEAT BASELINE!");
                }
            }
        }
        Commands::V9 => { experiments::v9_cosine_lr(device)?; }
        Commands::V10 => { experiments::v10_wider_model(device)?; }
        Commands::V11 => { experiments::v11_deeper_model(device)?; }
        Commands::V12 => { experiments::v12_efficient_attention(device)?; }
        Commands::V13 => { experiments::v13_token_augment(device)?; }
        Commands::V14 => { experiments::v14_adaptive_dropout(device)?; }
        Commands::Train { quick, max_lr, max_steps, warmup_pct, decay_pct, no_ema } => {
            use rust_experiments::{RealCosineLRExperiment, RealCosineLRConfig};
            let mut config = RealCosineLRConfig::default();
            config.max_lr = max_lr;
            config.learning_rate = max_lr as f64;
            config.max_steps = max_steps;
            config.wsd_warmup_pct = warmup_pct;
            config.wsd_decay_pct = decay_pct;
            config.use_ema = !no_ema;
            let exp = RealCosineLRExperiment::new(device).with_config(config);
            if quick {
                info!("Running quick test...");
                tokio::task::block_in_place(|| exp.quick_test())?;
            } else {
                info!("Running full WSD+EMA training...");
                tokio::task::block_in_place(|| exp.run())?;
            }
        }
        Commands::Infer {
            checkpoint, tokenizer, prompt, max_tokens,
            temperature, top_k, top_p, speculative, benchmark,
        } => {
            use rust_experiments::inference::{InferenceEngine, SamplingConfig};
            use rust_experiments::model::GPTConfig;

            let config = GPTConfig::default();
            let mut engine = InferenceEngine::from_checkpoint(
                &checkpoint, &tokenizer, config, device,
            )?;

            if speculative {
                engine = engine.with_speculative_decoding(3, 4);
                info!("Speculative decoding enabled (ngram=3, max_spec=4)");
            }

            let sampling = SamplingConfig {
                temperature, top_k, top_p, ..Default::default()
            };

            if benchmark {
                info!("Benchmarking inference...");
                let result = engine.benchmark(&prompt, max_tokens, &sampling)?;
                println!("\nBenchmark Results:");
                println!("  Prompt tokens: {}", result.prompt_tokens);
                println!("  Tokens generated: {}", result.tokens_generated);
                println!("  Prefill time: {:.2}ms", result.prefill_time_ms);
                println!("  Total time: {:.2}s", result.time_seconds);
                println!("  Tokens/sec: {:.2}", result.tokens_per_second);
            } else {
                info!("Generating: \"{}\"", prompt);
                let output = engine.generate(&prompt, max_tokens, &sampling)?;
                println!("\nPrompt: {}", prompt);
                println!("Output: {}", output);
            }
        }
    }

    Ok(())
}
