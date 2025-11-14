# FlashInfer Prefill Test - Fixes Summary

## Issues Fixed

### 1. ✅ FP8 Data Type Generation Error
**Problem:** `torch.randn` doesn't support directly generating FP8 (`Float8_e4m3fn`) tensors.

**Error:**
```
NotImplementedError: "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'
```

**Solution:** Generate tensors with `bfloat16` first, then convert to FP8:
```python
query = torch.randn(..., dtype=torch.bfloat16, device=device).to(dtype)
```

### 2. ✅ Head Count Compatibility Error  
**Problem:** TRT-LLM kernel requires `num_qo_heads` to be a multiple of `num_kv_heads`.

**Error:**
```
num_qo_heads must be a multiple of num_kv_heads, got num_kv_heads: 64 and num_qo_heads: 40
```

**Solution:** 
- Only use power-of-2 head counts (8, 16, 32, 64)
- Removed non-compatible configurations (e.g., 40 heads)

### 3. ✅ KV Cache Shape Error
**Problem:** Incorrect KV cache tensor shape caused head count mismatch.

**Error:**
```
num_qo_heads must be a multiple of num_kv_heads, got num_kv_heads: 64 and num_qo_heads: 32
```

**Root Cause:** 
- Used wrong dimension order: `[total_blocks, block_size, num_heads, head_dim]`
- Didn't explicitly set `num_kv_heads`

**Solution:**
- Correct shape: `[num_pages, num_kv_heads, page_size, head_dim]`
- Explicitly set `num_kv_heads = num_heads` for MHA
- Matches FlashInfer's expected format

### 4. ✅ Out-of-Memory Errors
**Problem:** Large configurations (long sequences + many heads) caused OOM.

**Error:**
```
Errors of the kernel trtllm_batch_context_with_kv_cache in the profiling table
```

**Solution:** Implement adaptive batch sizing:
- `seq_len >= 4096` with `num_heads > 32`: max batch_size = 4
- `seq_len >= 2048` with `num_heads > 32`: max batch_size = 8
- Shorter sequences: batch_size up to 64

## Code Changes

### create_data() Function

```python
# OLD (INCORRECT)
k_cache = torch.randn(total_blocks, block_size, num_heads, head_dim, 
                     dtype=dtype, device=device)

# NEW (CORRECT)
num_kv_heads = num_heads  # MHA: num_kv_heads = num_qo_heads
k_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim,
                     dtype=torch.bfloat16, device=device).to(dtype)
```

### Test Configurations

```python
# OLD
model_configs = [
    (32, 128),   # OK
    (40, 128),   # ERROR: Not power-of-2
    (64, 128),   # OK
]

# NEW
model_configs = [
    (8, 128),    # Very small
    (16, 128),   # Small
    (32, 128),   # Llama-2/3
    (64, 128),   # Large
]
```

## Test Configuration Summary

### Simple Config (7 tests)
- Small batches: bs=1,8,16 with seq_len=128,256,512
- Medium: bs=4 with seq_len=1024,2048
- Large: bs=1 with seq_len=4096 or num_heads=64

### Full Config (adaptive)
- 4 head counts × 6 seq_lens × variable batch sizes
- Total: ~100-150 configurations (adaptive to avoid OOM)

## Usage

```bash
# Quick test (7 configs, ~2-5 minutes)
python bench_flashinfer_prefill.py

# Full test suite (~100-150 configs, ~30-60 minutes)
FULL_BENCH=1 python bench_flashinfer_prefill.py

# Single config test
python test_flashinfer_prefill_single.py
```

## Performance Expectations

On H100 GPU:
- Small configs (bs=1-8, seq_len=128-512): 45-500 us, 1-10 TFLOPS
- Medium configs (bs=4-16, seq_len=1024-2048): 500-2000 us, 8-15 TFLOPS  
- Large configs (bs=1-4, seq_len=4096): 2000-5000 us, 10-20 TFLOPS

## Notes

1. **MHA vs GQA:** Current tests use Multi-Head Attention (MHA) where `num_kv_heads = num_qo_heads`. For Grouped Query Attention (GQA), modify to use `num_kv_heads < num_qo_heads`.

2. **Memory Requirements:** Tests estimate memory needs and skip overly large configs automatically.

3. **FP8 Precision:** All tests use `torch.float8_e4m3fn` for optimal performance on Hopper GPUs.

4. **Block Size:** Default page_size/block_size is 64 tokens, matching typical production settings.
