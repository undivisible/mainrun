#pragma once
#include <mlx/mlx.h>
#include <string>
#include <vector>

// Pre-tokenized BPE data loader. Returns mlx uint32 index tensors.
// JSON layout expected: {"train_ids":[...], "val_ids":[...], "val_char_count":N}
class DataLoader {
public:
  DataLoader(const std::string& data_path,
             const std::string& tokenizer_path,
             int batch_size = 32,
             int seq_len = 256);

  // Train batch: wraps around when exhausted. x,y are [batch_size, seq_len] uint32.
  std::pair<mlx::core::array, mlx::core::array> get_train_batch();

  // Non-overlapping val batch. Returns nullopt when exhausted.
  std::optional<std::pair<mlx::core::array, mlx::core::array>> next_val_batch();

  void reset_val_ptr() { val_ptr_ = 0; }

  int vocab_size() const { return vocab_size_; }
  int eos_id() const { return eos_id_; }
  long val_char_count() const { return val_char_count_; }
  const std::vector<uint32_t>& val_ids() const { return val_ids_; }
  int batch_size() const { return batch_size_; }
  int seq_len() const { return seq_len_; }

private:
  int batch_size_, seq_len_;
  std::vector<uint32_t> train_ids_, val_ids_;
  mlx::core::array train_arr_, val_arr_;
  size_t train_ptr_ = 0, val_ptr_ = 0;
  int vocab_size_ = 0, eos_id_ = 0;
  long val_char_count_ = 0;

  struct Parsed {
    std::vector<uint32_t> train_ids, val_ids;
    int vocab_size, eos_id;
    long val_char_count;
  };
  static Parsed load_data(const std::string& data_path, const std::string& tokenizer_path);
  DataLoader(Parsed&& p, int batch_size, int seq_len);

  std::pair<mlx::core::array, mlx::core::array>
  make_batch(const mlx::core::array& ids, size_t ptr);
};
