//! Real data loading from Hacker News dataset

use candle_core::{Device, Result, Tensor};
use tokenizers::{Tokenizer as HfTokenizer, models::Bpe, normalizers, pre_tokenizers, trainers, Decoder};
use anyhow::Context;

/// Real tokenizer using HuggingFace tokenizers
pub struct BPETokenizer {
    tokenizer: HfTokenizer,
    eos_token: String,
    eos_id: u32,
}

impl BPETokenizer {
    pub fn new(tokenizer: HfTokenizer, eos_token: &str) -> Result<Self> {
        let eos_id = tokenizer.token_to_id(eos_token).context("EOS token not found")?;
        Ok(Self {
            tokenizer,
            eos_token: eos_token.to_string(),
            eos_id,
        })
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false)
            .context("Failed to encode text")?;
        Ok(encoding.get_ids().to_vec())
    }
    
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let text = self.tokenizer.decode(ids, true)
            .context("Failed to decode ids")?;
        Ok(text)
    }
    
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size()
    }
    
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
}

/// Real data loader for Hacker News titles
pub struct RealDataLoader {
    device: Device,
    batch_size: usize,
    sequence_length: usize,
    train_ids: Vec<u32>,
    val_ids: Vec<u32>,
    train_ptr: usize,
    val_ptr: usize,
}

impl RealDataLoader {
    pub fn new(
        device: Device,
        batch_size: usize,
        sequence_length: usize,
        train_ids: Vec<u32>,
        val_ids: Vec<u32>,
    ) -> Self {
        Self {
            device,
            batch_size,
            sequence_length,
            train_ids,
            val_ids,
            train_ptr: 0,
            val_ptr: 0,
        }
    }
    
    /// Get a training batch
    pub fn get_train_batch(&mut self) -> Result<(Tensor, Tensor)> {
        let batch = self.get_batch_internal(&mut self.train_ptr, &self.train_ids)?;
        Ok(batch)
    }
    
    /// Get a validation batch
    pub fn get_val_batch(&mut self) -> Result<(Tensor, Tensor)> {
        let batch = self.get_batch_internal(&mut self.val_ptr, &self.val_ids)?;
        Ok(batch)
    }
    
    /// Get batch from data
    fn get_batch_internal(&self, ptr: &mut usize, data: &[u32]) -> Result<(Tensor, Tensor)> {
        let span = self.sequence_length * self.batch_size + 1;
        
        // Wrap around if we run out of data
        if *ptr + span >= data.len() {
            *ptr = 0;
        }
        
        let batch_data = &data[*ptr..*ptr + span];
        
        // Create input and target tensors
        let mut input_data = Vec::with_capacity(self.sequence_length * self.batch_size);
        let mut target_data = Vec::with_capacity(self.sequence_length * self.batch_size);
        
        for i in 0..self.sequence_length * self.batch_size {
            input_data.push(batch_data[i]);
            target_data.push(batch_data[i + 1]);
        }
        
        let input = Tensor::from_vec(
            input_data,
            (self.batch_size, self.sequence_length),
            &self.device,
        )?;
        
        let targets = Tensor::from_vec(
            target_data,
            (self.batch_size, self.sequence_length),
            &self.device,
        )?;
        
        *ptr += self.sequence_length * self.batch_size;
        Ok((input, targets))
    }
    
    /// Reset validation pointer for consistent evaluation
    pub fn reset_val_ptr(&mut self) {
        self.val_ptr = 0;
    }
    
    pub fn train_len(&self) -> usize {
        self.train_ids.len()
    }
    
    pub fn val_len(&self) -> usize {
        self.val_ids.len()
    }
}

/// Load and tokenize the Hacker News dataset
pub async fn load_hacker_news_data(
    vocab_size: usize,
    num_titles: usize,
    val_frac: f32,
    device: Device,
) -> anyhow::Result<(RealDataLoader, BPETokenizer)> {
    println!("Loading Hacker News dataset...");
    
    // For now, create mock data - in real implementation would load from HF datasets
    let (train_titles, val_titles) = create_mock_titles(num_titles, val_frac);
    
    println!("Training tokenizer on {} titles...", train_titles.len());
    
    // Train BPE tokenizer
    let tokenizer = train_bpe_tokenizer(&train_titles, vocab_size)?;
    let bpe_tokenizer = BPETokenizer::new(tokenizer, "<eos>")?;
    
    println!("Tokenizing training data...");
    
    // Tokenize the corpus
    let train_text = "<eos>".to_string() + &train_titles.join("<eos>") + "<eos>";
    let val_text = "<eos>".to_string() + &val_titles.join("<eos>") + "<eos>";
    
    let train_ids = bpe_tokenizer.encode(&train_text)?;
    let val_ids = bpe_tokenizer.encode(&val_text)?;
    
    println!("Dataset stats:");
    println!("  Train tokens: {}", train_ids.len());
    println!("  Val tokens: {}", val_ids.len());
    println!("  Vocab size: {}", bpe_tokenizer.vocab_size());
    
    // Create data loader
    let data_loader = RealDataLoader::new(
        device,
        4, // batch_size
        256, // sequence_length
        train_ids,
        val_ids,
    );
    
    Ok((data_loader, bpe_tokenizer))
}

/// Create mock titles for testing (replace with real HF dataset loading)
fn create_mock_titles(num_titles: usize, val_frac: f32) -> (Vec<String>, Vec<String>) {
    let mut titles = Vec::with_capacity(num_titles);
    
    // Create realistic Hacker News style titles
    let title_templates = vec![
        "Rust achieves {}% performance improvement over C++",
        "New AI model {} human-level reasoning",
        "Building {} with WebAssembly",
        "The {} guide to machine learning",
        "{}: A revolutionary approach to {}",
        "How {} companies are using {}",
        "The future of {} in {}",
        "{} years of {} development",
        "Why {} is the new {}",
        "Breaking: {} announces {}",
    ];
    
    let subjects = vec![
        "GPT", "Rust", "TypeScript", "Python", "JavaScript",
        "WebAssembly", "Docker", "Kubernetes", "React", "Vue",
        "PostgreSQL", "Redis", "GraphQL", "REST APIs", "Microservices",
        "Machine Learning", "Deep Learning", "Neural Networks", "AI", "Blockchain",
    ];
    
    for i in 0..num_titles {
        let template = &title_templates[i % title_templates.len()];
        let subject1 = &subjects[i % subjects.len()];
        let subject2 = &subjects[(i + 1) % subjects.len()];
        
        let title = if template.contains("{}{}") {
            template.replace("{}", "").replace("{}", &format!("{} {}", subject1, subject2))
        } else if template.matches("{}").count() == 2 {
            template.replacen("{}", subject1, 1).replacen("{}", subject2, 1)
        } else if template.matches("{}").count() == 1 {
            template.replacen("{}", subject1, 1)
        } else {
            template.to_string()
        };
        
        titles.push(title);
    }
    
    let split_idx = (num_titles as f32 * (1.0 - val_frac)) as usize;
    let train_titles = titles[..split_idx].to_vec();
    let val_titles = titles[split_idx..].to_vec();
    
    (train_titles, val_titles)
}

/// Train a BPE tokenizer on the given titles
fn train_bpe_tokenizer(titles: &[String], vocab_size: usize) -> anyhow::Result<HfTokenizer> {
    let mut tokenizer = HfTokenizer::new(Bpe::default());
    
    // Set up normalizer (same as Python version)
    let normalizer = normalizers::Sequence::new(vec![
        normalizers::NFKC,
        normalizers::Lowercase,
        normalizers::Replace(" ".into(), " ".into()),
        normalizers::Strip,
    ]);
    tokenizer.with_normalizer(normalizer);
    
    // Set up pre-tokenizer
    tokenizer.with_pre_tokenizer(pre_tokenizers::ByteLevel::new());
    
    // Set up decoder
    tokenizer.with_decoder(Decoder::ByteLevel::new());
    
    // Train tokenizer
    let special_tokens = vec!["<pad>", "<eos>", "<unk>"];
    let trainer = trainers::BpeTrainer::builder()
        .vocab_size(vocab_size)
        .special_tokens(special_tokens)
        .build();
    
    let mut texts: Vec<&str> = titles.iter().map(|s| s.as_str()).collect();
    tokenizer.train(&mut texts, trainer)?;
    
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_data_loading() {
        let device = Device::Cpu;
        let (mut data_loader, tokenizer) = load_hacker_news_data(1000, 0.1, device).await.unwrap();
        
        assert_eq!(tokenizer.vocab_size(), 1000);
        
        let (input, targets) = data_loader.get_train_batch().unwrap();
        assert_eq!(input.dims(), &[4, 256]);
        assert_eq!(targets.dims(), &[4, 256]);
        
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert!(!decoded.is_empty());
    }
}