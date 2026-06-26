//! Simple data loading without external tokenizer dependencies

use candle_core::{Device, Result, Tensor};


/// Simple tokenizer using character-level encoding for testing
pub struct SimpleTokenizer {
    vocab: Vec<char>,
    char_to_id: std::collections::HashMap<char, u32>,
    eos_token: char,
    eos_id: u32,
}

impl SimpleTokenizer {
    pub fn new(text: &str, vocab_size: usize, eos_token: char) -> Result<Self> {
        // Create character frequency map
        let mut freq = std::collections::HashMap::new();
        for ch in text.chars() {
            *freq.entry(ch).or_insert(0) += 1;
        }
        
        // Sort by frequency and take top vocab_size-1 chars
        let mut chars: Vec<_> = freq.into_iter().collect();
        chars.sort_by(|a, b| b.1.cmp(&a.1));
        
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut char_to_id = std::collections::HashMap::new();
        
        // Add most frequent characters
        for (ch, _) in chars.into_iter().take(vocab_size - 1) {
            char_to_id.insert(ch, vocab.len() as u32);
            vocab.push(ch);
        }
        
        // Add EOS token
        char_to_id.insert(eos_token, vocab.len() as u32);
        vocab.push(eos_token);
        
        let eos_id = char_to_id[&eos_token];
        
        Ok(Self {
            vocab,
            char_to_id,
            eos_token,
            eos_id,
        })
    }
    
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for ch in text.chars() {
            if let Some(&id) = self.char_to_id.get(&ch) {
                ids.push(id);
            } else {
                // Unknown character - use EOS token
                ids.push(self.eos_id);
            }
        }
        Ok(ids)
    }
    
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut text = String::new();
        for &id in ids {
            if let Some(&ch) = self.vocab.get(id as usize) {
                text.push(ch);
            }
        }
        Ok(text)
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
    
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
}

/// Simple data loader for text data
pub struct SimpleDataLoader {
    device: Device,
    batch_size: usize,
    sequence_length: usize,
    train_ids: Vec<u32>,
    val_ids: Vec<u32>,
    train_ptr: usize,
    val_ptr: usize,
}

impl SimpleDataLoader {
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
        let span = self.sequence_length * self.batch_size + 1;
        
        // Wrap around if we run out of data
        if self.train_ptr + span >= self.train_ids.len() {
            self.train_ptr = 0;
        }
        
        let batch_data = &self.train_ids[self.train_ptr..self.train_ptr + span];
        
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
        
        self.train_ptr += self.sequence_length * self.batch_size;
        Ok((input, targets))
    }
    
    /// Get a validation batch
    pub fn get_val_batch(&mut self) -> Result<(Tensor, Tensor)> {
        let span = self.sequence_length * self.batch_size + 1;
        
        // Wrap around if we run out of data
        if self.val_ptr + span >= self.val_ids.len() {
            self.val_ptr = 0;
        }
        
        let batch_data = &self.val_ids[self.val_ptr..self.val_ptr + span];
        
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
        
        self.val_ptr += self.sequence_length * self.batch_size;
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

/// Load and tokenize text data
pub async fn load_simple_data(
    vocab_size: usize,
    num_titles: usize,
    val_frac: f32,
    device: Device,
) -> anyhow::Result<(SimpleDataLoader, SimpleTokenizer)> {
    println!("Loading text dataset...");
    
    // Try to load real HN titles first, fall back to mock data
    let (train_titles, val_titles) = match load_real_hn_titles(num_titles, val_frac) {
        Ok(data) => {
            println!("Using real HN titles");
            data
        }
        Err(e) => {
            println!("Failed to load real HN titles ({}), using mock data", e);
            create_mock_titles(num_titles, val_frac)
        }
    };
    
    // Combine all text for tokenizer training
    let all_text = train_titles.join("") + &val_titles.join("");
    
    println!("Training tokenizer on {} characters...", all_text.len());
    
    // Train simple tokenizer
    let tokenizer = SimpleTokenizer::new(&all_text, vocab_size, '\n')?;
    
    println!("Tokenizing training data...");
    
    // Tokenize the corpus
    let train_text = train_titles.join("\n") + "\n";
    let val_text = val_titles.join("\n") + "\n";
    
    let train_ids = tokenizer.encode(&train_text)?;
    let val_ids = tokenizer.encode(&val_text)?;
    
    println!("Dataset stats:");
    println!("  Train tokens: {}", train_ids.len());
    println!("  Val tokens: {}", val_ids.len());
    println!("  Vocab size: {}", tokenizer.vocab_size());
    
    // Create data loader
    let data_loader = SimpleDataLoader::new(
        device,
        12, // batch_size - larger for better gradients
        256, // sequence_length
        train_ids,
        val_ids,
    );
    
    Ok((data_loader, tokenizer))
}

/// Load real HN titles from file
fn load_real_hn_titles(num_titles: usize, val_frac: f32) -> anyhow::Result<(Vec<String>, Vec<String>)> {
    let text = std::fs::read_to_string("hn_titles.txt")
        .or_else(|_| std::fs::read_to_string("../hn_titles.txt"))
        .or_else(|_| std::fs::read_to_string("../../hn_titles.txt"))
        .map_err(|e| anyhow::anyhow!("Cannot read hn_titles.txt: {}", e))?;
    
    let titles: Vec<String> = text.lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .take(num_titles)
        .collect();
    
    if titles.len() < 100 {
        return Err(anyhow::anyhow!("Not enough titles in file"));
    }
    
    let split_idx = (titles.len() as f32 * (1.0 - val_frac)) as usize;
    let train_titles = titles[..split_idx].to_vec();
    let val_titles = titles[split_idx..].to_vec();
    
    println!("  Loaded {} real HN titles (train: {}, val: {})", 
        train_titles.len() + val_titles.len(), train_titles.len(), val_titles.len());
    
    Ok((train_titles, val_titles))
}

/// Create mock titles for testing
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
        
        let title = if template.matches("{}").count() == 2 {
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_simple_data_loading() {
        let device = Device::Cpu;
        let (mut data_loader, tokenizer) = load_simple_data(1000, 1000, 0.1, device).await.unwrap();
        
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