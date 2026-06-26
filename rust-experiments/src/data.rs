//! Data loading and processing for experiments

use candle_core::{Device, Result, Tensor};
use rand::{SeedableRng, Rng};

/// Mock data loader for experiments
pub struct DataLoader {
    device: Device,
    batch_size: usize,
    sequence_length: usize,
    vocab_size: usize,
}

impl DataLoader {
    pub fn new(device: Device, batch_size: usize, sequence_length: usize, vocab_size: usize) -> Self {
        Self {
            device,
            batch_size,
            sequence_length,
            vocab_size,
        }
    }
    
    /// Get a random batch for training
    pub fn get_train_batch(&self) -> Result<(Tensor, Tensor)> {
        let total_tokens = self.batch_size * self.sequence_length;
        let tokens: Vec<u32> = (0..total_tokens)
            .map(|_| rand::random::<u32>() % self.vocab_size as u32)
            .collect();
        
        let input = Tensor::from_vec(tokens, (self.batch_size, self.sequence_length), &self.device)?;
        
        // Targets are input shifted by 1
        let input_vec = input.to_vec1::<u32>()?;
        let mut targets_tokens = input_vec[1..].to_vec();
        targets_tokens.push(input_vec[0]); // Wrap around
        
        let targets = Tensor::from_vec(targets_tokens, (self.batch_size, self.sequence_length), &self.device)?;
        
        Ok((input, targets))
    }
    
    /// Get a validation batch
    pub fn get_val_batch(&self) -> Result<(Tensor, Tensor)> {
        // Use a fixed seed for validation to ensure reproducibility
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let total_tokens = self.batch_size * self.sequence_length;
        let tokens: Vec<u32> = (0..total_tokens)
            .map(|_| rng.gen_range(0..self.vocab_size as u32))
            .collect();
        
        let input = Tensor::from_vec(tokens, (self.batch_size, self.sequence_length), &self.device)?;
        
        let input_vec = input.to_vec1::<u32>()?;
        let mut targets_tokens = input_vec[1..].to_vec();
        targets_tokens.push(input_vec[0]);
        
        let targets = Tensor::from_vec(targets_tokens, (self.batch_size, self.sequence_length), &self.device)?;
        
        Ok((input, targets))
    }
}

/// Tokenizer interface (mock implementation)
pub struct Tokenizer {
    vocab_size: usize,
}

impl Tokenizer {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Mock encoding - convert characters to token IDs
        let tokens: Vec<u32> = text
            .chars()
            .map(|c| (c as u32) % self.vocab_size as u32)
            .collect();
        Ok(tokens)
    }
    
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        // Mock decoding - convert token IDs back to characters
        let text: String = tokens
            .iter()
            .map(|&t| char::from_u32(t).unwrap_or('�'))
            .collect();
        Ok(text)
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_loader() {
        let device = Device::Cpu;
        let loader = DataLoader::new(device, 4, 128, 1000);
        
        let (input, targets) = loader.get_train_batch().unwrap();
        assert_eq!(input.dims(), &[4, 128]);
        assert_eq!(targets.dims(), &[4, 128]);
        
        // Check that targets are shifted version of input
        let input_flat = input.flatten_all().unwrap();
        let targets_flat = targets.flatten_all().unwrap();
        
        let input_vec = input_flat.to_vec1::<u32>().unwrap();
        let targets_vec = targets_flat.to_vec1::<u32>().unwrap();
        
        assert_eq!(targets_vec[0], input_vec[1]);
        assert_eq!(targets_vec[input_vec.len() - 1], input_vec[0]);
    }
    
    #[test]
    fn test_tokenizer() {
        let tokenizer = Tokenizer::new(1000);
        
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();
        
        assert_eq!(tokens.len(), text.len());
        // Note: Since this is a mock tokenizer, the decoded text won't match exactly
        assert_eq!(decoded.len(), text.len());
    }
}