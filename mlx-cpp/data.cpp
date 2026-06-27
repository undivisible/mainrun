#include "data.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <limits>

using namespace mlx::core;

static void validate_path(const std::string& path) {
  if (path.empty()) throw std::runtime_error("empty path");
  if (path.find('\0') != std::string::npos) throw std::runtime_error("path contains null byte");
  if (path.size() > 4096) throw std::runtime_error("path too long");
}

static int safe_size_t_to_int(size_t value) {
  if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("size too large for int conversion");
  return static_cast<int>(value);
}

// ponytail: minimal JSON extraction for the flat tokenized-data structure.
// Finds "key": [ints] / "key": number. No full JSON parser needed.
static std::string read_file(const std::string& path) {
  validate_path(path);
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static size_t find_key(const std::string& s, const std::string& key) {
  auto q = "\"" + key + "\"";
  auto p = s.find(q);
  if (p == std::string::npos)
    throw std::runtime_error("missing key " + key);
  return p + q.size();
}

static std::vector<uint32_t> parse_int_array(const std::string& s, const std::string& key) {
  auto p = find_key(s, key);
  auto lb = s.find('[', p);
  if (lb == std::string::npos) throw std::runtime_error("expected [ after " + key);
  std::vector<uint32_t> out;
  size_t i = lb + 1;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ' ' || s[i] == ',' || s[i] == '\n' ||
                            s[i] == '\r' || s[i] == '\t'))
      i++;
    if (i < s.size() && s[i] == ']') break;
    size_t j = i;
    while (j < s.size() && s[j] != ',' && s[j] != ']' &&
           s[j] != ' ' && s[j] != '\n')
      j++;
    out.push_back((uint32_t)std::stoul(s.substr(i, j - i)));
    i = j;
  }
  return out;
}

static long parse_long(const std::string& s, const std::string& key) {
  auto p = find_key(s, key);
  auto colon = s.find(':', p);
  if (colon == std::string::npos) throw std::runtime_error("expected : after " + key);
  size_t i = colon + 1;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n')) i++;
  size_t j = i;
  while (j < s.size() && (s[j] == '-' || (s[j] >= '0' && s[j] <= '9'))) j++;
  return std::stol(s.substr(i, j - i));
}

// ponytail: tokenizer JSON only needs vocab_size (count vocab entries) and eos_id.
// Scan the "vocab":{...} block with string-aware brace tracking.
static void parse_tokenizer(const std::string& s, int& vocab_size, int& eos_id) {
  auto vpos = s.find("\"vocab\"");
  if (vpos == std::string::npos) throw std::runtime_error("no vocab in tokenizer");
  auto lb = s.find('{', vpos);
  int depth = 0;
  bool in_str = false, esc = false;
  int entries = 0;
  bool counted_key = false;
  eos_id = -1;
  size_t i = lb;
  for (; i < s.size(); i++) {
    char c = s[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') {
      in_str = true;
      // detect "<eos>" key
      if (depth == 1 && i + 6 <= s.size() && s.compare(i, 6, "\"<eos>\"") == 0) {
        auto colon = s.find(':', i + 6);
        if (colon == std::string::npos || colon + 1 >= s.size())
          throw std::runtime_error("invalid eos token format");
        eos_id = std::stoi(s.substr(colon + 1));
      }
      if (depth == 1 && !counted_key) { entries++; counted_key = true; }
      continue;
    }
    if (c == '{') { depth++; }
    else if (c == '}') {
      depth--;
      if (depth == 0) break;
    } else if (c == ',' && depth == 1) {
      counted_key = false;
    }
  }
  vocab_size = entries;
  if (eos_id < 0) eos_id = 1;
}

DataLoader::DataLoader(const std::string& data_path,
                       const std::string& tokenizer_path,
                       int batch_size, int seq_len)
    : DataLoader(load_data(data_path, tokenizer_path), batch_size, seq_len) {}

DataLoader::Parsed DataLoader::load_data(const std::string& data_path,
                                          const std::string& tokenizer_path) {
  std::string ds = read_file(data_path);
  Parsed p;
  p.train_ids = parse_int_array(ds, "train_ids");
  p.val_ids = parse_int_array(ds, "val_ids");
  p.val_char_count = parse_long(ds, "val_char_count");
  std::string ts = read_file(tokenizer_path);
  parse_tokenizer(ts, p.vocab_size, p.eos_id);
  fprintf(stderr, "Dataset: train=%zu val=%zu vocab=%d eos=%d val_chars=%ld\n",
          p.train_ids.size(), p.val_ids.size(), p.vocab_size, p.eos_id, p.val_char_count);
  return p;
}

DataLoader::DataLoader(Parsed&& p, int batch_size, int seq_len)
    : batch_size_(batch_size), seq_len_(seq_len),
      train_ids_(std::move(p.train_ids)),
      val_ids_(std::move(p.val_ids)),
      train_arr_(train_ids_.begin(), {safe_size_t_to_int(train_ids_.size())}, uint32),
      val_arr_(val_ids_.begin(), {safe_size_t_to_int(val_ids_.size())}, uint32),
      vocab_size_(p.vocab_size), eos_id_(p.eos_id),
      val_char_count_(p.val_char_count) {}

static int safe_span(int seq_len, int batch_size) {
  if (seq_len <= 0 || batch_size <= 0) return 1;
  if (seq_len > std::numeric_limits<int>::max() / batch_size)
    throw std::runtime_error("batch dimensions too large");
  return seq_len * batch_size + 1;
}

std::pair<array, array> DataLoader::make_batch(const array& ids, size_t ptr) {
  int span = safe_span(seq_len_, batch_size_);
  int ptr_i = safe_size_t_to_int(ptr);
  if (ptr_i > std::numeric_limits<int>::max() - span)
    throw std::runtime_error("batch offset overflow");
  auto batch = slice(ids, {ptr_i}, {ptr_i + span});
  auto x = reshape(slice(batch, {0}, {span - 1}), {batch_size_, seq_len_});
  auto y = reshape(slice(batch, {1}, {span}), {batch_size_, seq_len_});
  return {x, y};
}

std::pair<array, array> DataLoader::get_train_batch() {
  int span = safe_span(seq_len_, batch_size_);
  if (train_ptr_ + static_cast<size_t>(span) >= train_ids_.size()) train_ptr_ = 0;
  auto b = make_batch(train_arr_, train_ptr_);
  train_ptr_ += seq_len_ * batch_size_;
  return b;
}

std::optional<std::pair<array, array>> DataLoader::next_val_batch() {
  int span = safe_span(seq_len_, batch_size_);
  if (val_ptr_ + static_cast<size_t>(span) > val_ids_.size()) return std::nullopt;
  auto b = make_batch(val_arr_, val_ptr_);
  val_ptr_ += span;
  return b;
}
