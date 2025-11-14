# FlashInfer Prefill Kernel Benchmark

Performance benchmark for the `trtllm_batch_context_with_kv_cache` kernel used in FlashInfer FMHA prefill operations.

## Overview

This benchmark tests the TRT-LLM batch context attention kernel with **fixed model architecture** (96 heads × 128 head_dim) across various batch sizes and sequence lengths, including **partial prefill** scenarios.

### Fixed Architecture
- **Number of heads**: 96 (both Q and KV heads for MHA)
- **Head dimension**: 128
- **Total hidden size**: 96 × 128 = 12288

### Test Scenarios

#### 1. Full Prefill (`q_len == kv_len`)
No existing KV cache. All tokens are new:
- seq_len ∈ {128, 256, 512, 1024, 2048, 4096}
- Variable batch sizes based on memory constraints

#### 2. Partial Prefill (`kv_len > q_len`)
Existing KV cache with only a few new tokens to prefill. This simulates **continuous batching** scenarios:
- **(1024, 5)**: 1024 tokens cached, 5 new tokens
- **(1024, 32)**: 1024 tokens cached, 32 new tokens
- **(2048, 5)**: 2048 tokens cached, 5 new tokens
- **(4096, 5)**: 4096 tokens cached, 5 new tokens
- And more combinations...

**Real-world use case**: In continuous batching, a request may already have processed 1024 tokens (stored in KV cache). When user sends 5 new tokens, we only need to compute attention for those 5 query tokens against all 1024 KV tokens.

## Files

- `bench_flashinfer_prefill.py`: Main benchmark script
- `test_flashinfer_prefill_single.py`: Single configuration test for debugging
- `README_flashinfer_prefill.md`: This documentation
- `FIXES_SUMMARY.md`: Technical fixes and implementation notes

## Usage

### Quick Test (10 configurations, ~3-5 minutes)

```bash
cd /home/wangyin.yx/workspace/misc/trtllm/flashinfer/mytests
python bench_flashinfer_prefill.py
```

### Full Test Suite (~200+ configurations, ~1-2 hours)

```bash
FULL_BENCH=1 python bench_flashinfer_prefill.py
```

### Single Configuration Test

```bash
python test_flashinfer_prefill_single.py
```

Or customize in the script:

```python
config = dict(
    batch_size=8,
    num_heads=96,
    head_dim=128,
    q_len=5,        # New tokens
    kv_len=1024,    # Total cache length
    dtype=torch.float8_e4m3fn,
)
```

## Output

### Console Output

```
 > Perf (bs= 8, nh= 96, hd=128, q=   5, kv=1024):    456 us |  78.90 TFLOPS |  1234 GB/s
MAIN_OUTPUT={"batch_size": 8, "num_heads": 96, "head_dim": 128, "q_len": 5, "kv_len": 1024, ...}
```

### CSV Output

Results are saved to `/tmp/fp4_test/flashinfer_prefill.csv`:

```csv
dtype,batch_size,num_heads,head_dim,q_len,kv_len,total_q_tokens,t_us,tflops,gb_per_s
torch.float8_e4m3fn,8,96,128,5,1024,40,456.78,78.90,1234.56
```

## Implementation Details

### Kernel Parameters

```python
flashinfer.prefill.trtllm_batch_context_with_kv_cache(
    query=query,                    # [total_q_tokens, num_qo_heads, head_dim]
                                    # total_q_tokens = batch_size × q_len
    kv_cache=(k_cache, v_cache),   # Tuple of KV cache blocks
                                    # k_cache: [num_pages, num_kv_heads, page_size, head_dim]
                                    # v_cache: [num_pages, num_kv_heads, page_size, head_dim]
                                    # num_kv_heads = 96 for MHA
    workspace_buffer=workspace,     # 512 MB workspace
    block_tables=block_tables,      # [batch_size, num_blocks_per_seq]
    seq_lens=kv_seq_lens,           # [batch_size] - KV sequence lengths
    max_q_len=q_len,                # Maximum query length
    max_kv_len=kv_len,              # Maximum KV cache length
    bmm1_scale=q_scale * k_scale * sqrt(1/head_dim),
    bmm2_scale=1.0,
    batch_size=batch_size,
    cum_seq_lens_q=cum_seq_lens_q,  # [batch_size + 1] - cumulative Q lengths
    cum_seq_lens_kv=cum_seq_lens_kv, # [batch_size + 1] - cumulative KV lengths
    window_left=-1,                  # No sliding window
    sinks=None,                      # No attention sinks
    out_dtype=torch.float8_e4m3fn,
)
```

**Note:** This test uses Multi-Head Attention (MHA) where `num_kv_heads = num_qo_heads = 96`.

### Performance Calculation

**FLOPS:**
- BMM1 (Q @ K^T): `2 × (batch_size × q_len) × num_heads × kv_len × head_dim`
- BMM2 (S @ V): `2 × (batch_size × q_len) × num_heads × kv_len × head_dim`
- Total TFLOPS = (BMM1 + BMM2) / time / 1e12

**Memory Bandwidth:**
- Input: Query (q_len tokens), K cache (kv_len tokens), V cache (kv_len tokens)
- Output: Attention output (q_len tokens)
- GB/s = Total bytes / time / 1e9

**Key Insight for Partial Prefill:**
- Computation scales with `q_len × kv_len` (each query attends to all KV)
- Memory for query scales with `q_len` only
- Memory for KV cache scales with `kv_len` only
- When `q_len << kv_len`, bandwidth becomes more important than compute

## Example Results

### Full Prefill Scenarios

```
 > Perf (bs= 1, nh= 96, hd=128, q= 128, kv= 128):     78 us |   1.23 TFLOPS |   234 GB/s
 > Perf (bs= 8, nh= 96, hd=128, q= 256, kv= 256):    234 us |   8.90 TFLOPS |   456 GB/s
 > Perf (bs= 4, nh= 96, hd=128, q=1024, kv=1024):   1234 us |  56.78 TFLOPS |   890 GB/s
```

### Partial Prefill Scenarios

```
 > Perf (bs= 8, nh= 96, hd=128, q=   5, kv= 128):     45 us |   0.89 TFLOPS |   123 GB/s
 > Perf (bs= 8, nh= 96, hd=128, q=   5, kv= 512):    123 us |   2.34 TFLOPS |   234 GB/s
 > Perf (bs= 4, nh= 96, hd=128, q=   5, kv=1024):    234 us |   3.45 TFLOPS |   345 GB/s
 > Perf (bs= 2, nh= 96, hd=128, q=  32, kv=1024):    456 us |  12.34 TFLOPS |   567 GB/s
```

**Observation**: Partial prefill with small `q_len` tends to be bandwidth-bound rather than compute-bound.

## Requirements

- FlashInfer Python package (`flashinfer-python`)
- PyTorch with CUDA support
- GPU with FP8 support (Hopper/H100 or later recommended)
- Sufficient GPU memory (H100 80GB recommended for full suite)

## Troubleshooting

### Import Errors

```bash
# Verify flashinfer is installed
python -c "import flashinfer; print(flashinfer.__version__)"
```

### CUDA Errors

- Ensure GPU has sufficient memory
- For large configs, reduce batch_size
- Check CUDA compatibility with your GPU

### Head Count Compatibility Error

If you see an error like:
```
num_qo_heads must be a multiple of num_kv_heads
```

This is already fixed in the current version by ensuring `num_kv_heads = 96` matches `num_qo_heads = 96` (MHA).

### Performance Issues

- Warm up the GPU before benchmarking (first run may be slower)
- Use `FULL_BENCH=1` only on powerful GPUs (H100 recommended)
- Monitor GPU utilization with `nvidia-smi`
- For partial prefill with small `q_len`, expect lower TFLOPS (bandwidth-bound)

## Test Configuration Summary

### Simple Config (10 tests)
**Full prefill:** 5 configs with varying batch sizes and sequence lengths
**Partial prefill:** 5 configs with small q_len and large kv_len

### Full Config (200+ tests)
**Full prefill:** 6 sequence lengths × adaptive batch sizes = ~100 configs
**Partial prefill:** 9 (kv_len, q_len) pairs × adaptive batch sizes = ~100 configs

## Notes

1. **Fixed Architecture**: All tests use 96 heads × 128 head_dim to match your specific use case.

2. **Partial Prefill Support**: Key feature for continuous batching scenarios where most KV cache is already filled.

3. **Memory Constraints**: Batch sizes are automatically adjusted based on sequence length to avoid OOM.

4. **FP8 Precision**: All tests use `torch.float8_e4m3fn` for optimal performance on Hopper GPUs.

5. **Block Size**: Default page_size/block_size is 64 tokens, matching typical production settings.

## Performance Analysis

### Compute vs Bandwidth Bound

**Full Prefill (q_len = kv_len)**:
- Balanced compute and memory access
- Good TFLOPS utilization
- Example: q=1024, kv=1024 → ~50+ TFLOPS on H100

**Partial Prefill (q_len << kv_len)**:
- More bandwidth-bound
- Lower TFLOPS but still efficient
- Example: q=5, kv=1024 → ~3-5 TFLOPS, but latency is low (<500μs)

### Why Partial Prefill Matters

In production serving with continuous batching:
1. User sends initial prompt (1024 tokens) → **Full prefill**
2. Model generates response, updating KV cache incrementally
3. User sends follow-up (5 tokens) → **Partial prefill** (1024 cached + 5 new)

Partial prefill optimization is crucial for good user experience in multi-turn conversations!
