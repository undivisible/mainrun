//! BPE tokenizer data loading from pre-tokenized data

use candle_core::{Device, Result, Tensor};
use tokenizers::Tokenizer as HfTokenizer;
use std::path::Path;

/// BPE tokenizer wrapper that loads a pre-trained tokenizer
pub struct BpeTokenizer {
    tokenizer: HfTokenizer,
    vocab_size: usize,
    eos_id: u32,
}

impl BpeTokenizer {
    /// Load a pre-trained tokenizer from file
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let tokenizer = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        let vocab_size = tokenizer.get_vocab_size(false);
        let eos_id = tokenizer.token_to_id("<eos>").unwrap_or(0);
        Ok(Self { tokenizer, vocab_size, eos_id })
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenize error: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }
    
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| candle_core::Error::Msg(format!("Decode error: {}", e)))
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
}

/// Data loader using pre-tokenized BPE data
pub struct BpeDataLoader {
    device: Device,
    batch_size: usize,
    sequence_length: usize,
    train_ids: Vec<u32>,
    val_ids: Vec<u32>,
    train_ptr: usize,
    val_ptr: usize,
}

impl BpeDataLoader {
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
    
    pub fn get_train_batch(&mut self) -> Result<(Tensor, Tensor)> {
        let span = self.sequence_length * self.batch_size + 1;
        
        if self.train_ptr + span >= self.train_ids.len() {
            self.train_ptr = 0;
        }
        
        let batch_data = &self.train_ids[self.train_ptr..self.train_ptr + span];
        
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
        
        self.train_ptr += self.sequence_length * self.batch_size;
        Ok((input, targets))
    }
    
    pub fn get_val_batch(&mut self) -> Result<(Tensor, Tensor)> {
        let span = self.sequence_length * self.batch_size + 1;
        
        if self.val_ptr + span >= self.val_ids.len() {
            self.val_ptr = 0;
        }
        
        let batch_data = &self.val_ids[self.val_ptr..self.val_ptr + span];
        
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
        
        self.val_ptr += self.sequence_length * self.batch_size;
        Ok((input, targets))
    }
    
    pub fn reset_val_ptr(&mut self) {
        self.val_ptr = 0;
    }

    pub fn train_ids(&self) -> &[u32] {
        &self.train_ids
    }

    pub fn val_ids(&self) -> &[u32] {
        &self.val_ids
    }

    pub fn batch_size(&self) -> usize { self.batch_size }
    pub fn seq_length(&self) -> usize { self.sequence_length }
}

/// Load pre-tokenized BPE data
pub async fn load_bpe_data(
    _vocab_size: usize,
    _num_titles: usize,
    _val_frac: f32,
    device: Device,
) -> anyhow::Result<(BpeDataLoader, BpeTokenizer)> {
    println!("Loading pre-tokenized BPE data...");
    
    // Find the tokenized data file
    let data_path = find_data_file("tokenized_data.json")
        .ok_or_else(|| anyhow::anyhow!("Cannot find tokenized_data.json"))?;
    let tokenizer_path = find_data_file("tokenizer.json")
        .ok_or_else(|| anyhow::anyhow!("Cannot find tokenizer.json"))?;
    
    // Load tokenizer
    let tokenizer = BpeTokenizer::from_file(&tokenizer_path)?;
    println!("  Tokenizer vocab size: {}", tokenizer.vocab_size());
    
    // Load pre-tokenized data
    let data_str = std::fs::read_to_string(&data_path)?;
    let data: serde_json::Value = serde_json::from_str(&data_str)?;
    
    let train_ids: Vec<u32> = serde_json::from_value(data["train_ids"].clone())?;
    let val_ids: Vec<u32> = serde_json::from_value(data["val_ids"].clone())?;
    
    println!("Dataset stats:");
    println!("  Train tokens: {}", train_ids.len());
    println!("  Val tokens: {}", val_ids.len());
    println!("  Vocab size: {}", tokenizer.vocab_size());
    
    let data_loader = BpeDataLoader::new(
        device,
        32, // batch_size - match Python
        256, // sequence_length
        train_ids,
        val_ids,
    );
    
    Ok((data_loader, tokenizer))
}

/// Find a data file by searching in common locations
fn find_data_file(filename: &str) -> Option<String> {
    let candidates = [
        filename.to_string(),
        format!("../{}", filename),
        format!("../../{}", filename),
        format!("../../../{}", filename),
    ];
    
    for path in &candidates {
        if Path::new(path).exists() {
            return Some(path.clone());
        }
    }
    None
}
