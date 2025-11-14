# FlashInfer Prefill Kernel Benchmark

Performance benchmark for the `trtllm_batch_context_with_kv_cache` kernel used in FlashInfer FMHA prefill operations.

## Files

- `bench_flashinfer_prefill.py`: Main benchmark script
- `test_flashinfer_prefill_single.py`: Single configuration test for quick debugging

## Overview

This benchmark tests the TRT-LLM batch context attention kernel with various configurations:

### Test Configurations

**Model Architectures:**
- Llama-2/3 style: 32 heads × 128 head_dim
- Qwen style: 40 heads × 128 head_dim
- Large models: 64 heads × 128 head_dim

**Batch Sizes:** 1, 2, 4, 8, 16, 32, 64

**Sequence Lengths:** 128, 256, 512, 1024, 2048, 4096

### Performance Metrics

- **Execution Time** (microseconds)
- **TFLOPS**: Floating-point operations per second
- **GB/s**: Memory bandwidth utilization

## Usage

### Quick Test (6 configurations)

```bash
cd /home/wangyin.yx/workspace/misc/trtllm/flashinfer/mytests
python bench_flashinfer_prefill.py
```

### Full Test Suite (126 configurations)

```bash
FULL_BENCH=1 python bench_flashinfer_prefill.py
```

### Single Configuration Test

```bash
python test_flashinfer_prefill_single.py
```

Or customize the configuration in the script:

```python
config = dict(
    batch_size=16,
    num_heads=32,
    head_dim=128,
    seq_len=1024,
    max_kv_len=1024,
    dtype=torch.float8_e4m3fn,
)
```

## Output

### Console Output

```
 > Perf (bs= 8, nh= 32, hd=128, sl= 256, max_kv= 256):    123 us |   4.56 TFLOPS |   234 GB/s
MAIN_OUTPUT={"batch_size": 8, "num_heads": 32, "head_dim": 128, "seq_len": 256, ...}
```

### CSV Output

Results are saved to `/tmp/fp4_test/flashinfer_prefill.csv`:

```csv
dtype,batch_size,num_heads,head_dim,seq_len,max_kv_len,total_tokens,t_us,tflops,gb_per_s
torch.float8_e4m3fn,8,32,128,256,256,2048,123.45,4.56,234.78
```

## Implementation Details

### Kernel Parameters

```python
flashinfer.prefill.trtllm_batch_context_with_kv_cache(
    query=query,                    # [total_tokens, num_heads, head_dim]
    kv_cache=(k_cache, v_cache),   # KV cache blocks [num_blocks, block_size, nh, hd]
    workspace_buffer=workspace,     # 512 MB workspace
    block_tables=block_tables,      # [batch_size, num_blocks_per_seq]
    seq_lens=seq_lens,              # [batch_size]
    max_q_len=max_seq_len,
    max_kv_len=max_kv_len,
    bmm1_scale=q_scale * k_scale * sqrt(1/head_dim),
    bmm2_scale=1.0,
    batch_size=batch_size,
    cum_seq_lens_q=cum_seq_lens,    # [batch_size + 1]
    cum_seq_lens_kv=cum_seq_lens,
    window_left=-1,                  # No sliding window
    sinks=None,                      # No attention sinks
    out_dtype=torch.float8_e4m3fn,
)
```

### Performance Calculation

**FLOPS:**
- BMM1 (Q @ K^T): `2 × total_tokens × num_heads × seq_len × head_dim`
- BMM2 (S @ V): `2 × total_tokens × num_heads × seq_len × head_dim`
- Total TFLOPS = (BMM1 + BMM2) / time / 1e12

**Memory Bandwidth:**
- Input: Query, K cache, V cache
- Output: Attention output
- GB/s = Total bytes / time / 1e9

## Example Results

Typical performance on H100:

```
 > Perf (bs= 1, nh= 32, hd=128, sl= 128, max_kv= 128):     45 us |   1.12 TFLOPS |   156 GB/s
 > Perf (bs= 8, nh= 32, hd=128, sl= 256, max_kv= 256):    123 us |   4.56 TFLOPS |   234 GB/s
 > Perf (bs=16, nh= 32, hd=128, sl= 512, max_kv= 512):    456 us |   8.92 TFLOPS |   312 GB/s
 > Perf (bs= 4, nh= 40, hd=128, sl=1024, max_kv=1024):   1234 us |  12.34 TFLOPS |   456 GB/s
```

## Comparison with Other Tests

This benchmark follows the same pattern as other tests in this directory:

- Uses `flashinfer.testing.utils.bench_kineto` for performance measurement
- Reports TFLOPS and GB/s metrics
- Saves results to CSV
- Outputs JSON metrics (MAIN_OUTPUT)
- Supports both quick and comprehensive test suites

**Key Differences:**
- Tests attention kernel instead of GEMM operations
- Uses block-based KV cache structure
- FP8 data type optimized for modern GPUs (Hopper+)
- Variable-length sequences with cumulative length tracking

## Requirements

- FlashInfer Python package (`flashinfer-python`)
- PyTorch with CUDA support
- GPU with FP8 support (Hopper/H100 or later recommended)

## Troubleshooting

### Import Errors

```bash
# Verify flashinfer is installed
python -c "import flashinfer; print(flashinfer.__version__)"
```

### CUDA Errors

- Ensure GPU has sufficient memory
- For large configs, reduce batch_size or seq_len
- Check CUDA compatibility with your GPU

### Performance Issues

- Warm up the GPU before benchmarking (first run may be slower)
- Use `FULL_BENCH=1` only on powerful GPUs
- Monitor GPU utilization with `nvidia-smi`

