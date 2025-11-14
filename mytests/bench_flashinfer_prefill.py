"""
Benchmark for FlashInfer prefill kernel: trtllm_batch_context_with_kv_cache

This test benchmarks the TRT-LLM batch context attention kernel used in FlashInfer FMHA.
It measures performance metrics (latency, TFLOPS, bandwidth) for various batch sizes,
sequence lengths, and model configurations.
"""

import json
import os
import torch

try:
    import flashinfer
    from flashinfer.testing.utils import bench_kineto, count_bytes
except ImportError:
    print("Skipping bench_flashinfer_prefill.py since required modules are not available")
    exit(0)


def bench_one(
    batch_size,
    num_heads,
    head_dim,
    seq_len,
    max_kv_len,
    block_size=64,
    dtype=torch.float8_e4m3fn,
):
    """
    Benchmark trtllm_batch_context_with_kv_cache kernel

    Args:
        batch_size: Number of sequences in the batch
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        seq_len: Query sequence length per sample
        max_kv_len: Maximum KV cache length
        block_size: KV cache block size
        dtype: Data type for computation
    """

    data = create_data(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        max_kv_len=max_kv_len,
        block_size=block_size,
        dtype=dtype,
    )

    def test_func():
        o = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=data["query"],
            kv_cache=(data["k_cache"], data["v_cache"]),
            workspace_buffer=data["workspace_buffer"],
            block_tables=data["block_tables"],
            seq_lens=data["seq_lens"],
            max_q_len=data["max_q_len"],
            max_kv_len=data["max_kv_len"],
            bmm1_scale=data["bmm1_scale"],
            bmm2_scale=data["bmm2_scale"],
            batch_size=data["batch_size"],
            cum_seq_lens_q=data["cum_seq_lens_q"],
            cum_seq_lens_kv=data["cum_seq_lens_kv"],
            window_left=-1,
            sinks=None,
            out_dtype=dtype,
        )

    # Benchmark using kineto
    t = bench_kineto(
        test_func,
        "trtllm_batch_context_with_kv_cache",
        suppress_kineto_output=True,
        num_tests=5,
    )

    # Calculate metrics
    total_tokens = data["seq_lens"].sum().item()

    # FLOPS calculation for attention: 2 * total_tokens * seq_len * num_heads * head_dim
    # BMM1: Q @ K^T = [total_tokens, num_heads, head_dim] @ [num_heads, head_dim, seq_len]
    #       -> [total_tokens, num_heads, seq_len]
    # BMM2: S @ V = [total_tokens, num_heads, seq_len] @ [num_heads, seq_len, head_dim]
    #       -> [total_tokens, num_heads, head_dim]
    flops_bmm1 = 2 * total_tokens * num_heads * seq_len * head_dim
    flops_bmm2 = 2 * total_tokens * num_heads * seq_len * head_dim
    total_flops = flops_bmm1 + flops_bmm2
    tflops = total_flops / t / 1e12

    # Memory bandwidth calculation
    # Input: Q, K, V
    # Output: O
    query_bytes = data["query"].numel() * data["query"].element_size()
    kv_bytes = (data["k_cache"].numel() + data["v_cache"].numel()) * data["k_cache"].element_size()
    output_bytes = total_tokens * num_heads * head_dim * torch.finfo(dtype).bits // 8
    total_bytes = query_bytes + kv_bytes + output_bytes
    gb_per_s = total_bytes / 1e9 / t

    print(
        f" > Perf (bs={batch_size:2}, nh={num_heads:3}, hd={head_dim:3}, sl={seq_len:4}, max_kv={max_kv_len:4}): "
        f"{t * 1e6:6.0f} us | {tflops:6.2f} TFLOPS | {gb_per_s:6.0f} GB/s"
    )

    # Save results to CSV
    output_dir = "/tmp/fp4_test"
    os.makedirs(output_dir, exist_ok=True)
    f = open(f"{output_dir}/flashinfer_prefill.csv", "a")
    f.write(
        f"{dtype},{batch_size},{num_heads},{head_dim},{seq_len},{max_kv_len},"
        f"{total_tokens},{t*1e6},{tflops},{gb_per_s}\n"
    )
    f.close()

    metrics = dict(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        max_kv_len=max_kv_len,
        total_tokens=total_tokens,
        t_us=t * 1e6,
        tflops=tflops,
        gb_per_s=gb_per_s,
    )
    print(f"MAIN_OUTPUT={json.dumps(metrics)}")


def create_data(
    batch_size,
    num_heads,
    head_dim,
    seq_len,
    max_kv_len,
    block_size=64,
    dtype=torch.float8_e4m3fn,
    device="cuda:0",
):
    """
    Create test data for FlashInfer prefill kernel

    Returns a dictionary with:
        - query: [total_tokens, num_heads, head_dim]
        - k_cache, v_cache: KV cache blocks
        - workspace_buffer: workspace memory
        - block_tables: [batch_size, max_blocks]
        - seq_lens: [batch_size]
        - cum_seq_lens_q, cum_seq_lens_kv: cumulative sequence lengths
        - bmm1_scale, bmm2_scale: scaling factors
        - other metadata
    """
    device = torch.device(device)

    # Create query tensor: [total_tokens, num_heads, head_dim]
    # Note: torch.randn doesn't support FP8, so we create with bfloat16 then convert
    total_tokens = batch_size * seq_len
    query = torch.randn(total_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device).to(dtype)

    # Create sequence lengths (all same for simplicity)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    # Calculate cumulative sequence lengths
    cum_seq_lens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cum_seq_lens_q[1:] = torch.cumsum(seq_lens, dim=0)
    cum_seq_lens_kv = cum_seq_lens_q.clone()

    # Create KV cache blocks
    # Calculate number of blocks needed
    num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # KV cache shape: [total_blocks, block_size, num_heads, head_dim]
    # Note: torch.randn doesn't support FP8, so we create with bfloat16 then convert
    k_cache = torch.randn(total_blocks, block_size, num_heads, head_dim, dtype=torch.bfloat16, device=device).to(dtype)
    v_cache = torch.randn(total_blocks, block_size, num_heads, head_dim, dtype=torch.bfloat16, device=device).to(dtype)

    # Create block tables: [batch_size, max_blocks_per_seq]
    block_tables = torch.arange(total_blocks, dtype=torch.int32, device=device).reshape(
        batch_size, num_blocks_per_seq
    )

    # Create workspace buffer (512 MB)
    workspace_buffer = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=device)

    # Scaling factors
    scaling = head_dim ** -0.5
    q_scale = 1.0
    k_scale = 1.0
    bmm1_scale = q_scale * k_scale * scaling
    bmm2_scale = 1.0

    return dict(
        query=query,
        k_cache=k_cache,
        v_cache=v_cache,
        workspace_buffer=workspace_buffer,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=seq_len,
        max_kv_len=max_kv_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        batch_size=batch_size,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
    )


def enumerate_test_configs():
    """
    Generate test configurations similar to real-world inference scenarios

    Note: Only use power-of-2 head counts to avoid GQA compatibility issues.
    The TRT-LLM kernel requires num_qo_heads to be a multiple of num_kv_heads.
    """
    # Common model configurations (power of 2 for compatibility)
    model_configs = [
        # (num_heads, head_dim)
        (32, 128),   # Llama-2/3 style, GPT-3
        (64, 128),   # Larger models
        (16, 128),   # Smaller models
        (8, 128),    # Very small models
    ]

    # Batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    # Sequence lengths (typical prefill lengths)
    seq_lens = [128, 256, 512, 1024, 2048, 4096]

    # Generate configurations
    for num_heads, head_dim in model_configs:
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                # max_kv_len typically same as seq_len for prefill
                max_kv_len = seq_len
                yield dict(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    seq_len=seq_len,
                    max_kv_len=max_kv_len,
                    dtype=torch.float8_e4m3fn,
                )


def enumerate_simple_configs():
    """
    Generate a smaller set of test configurations for quick testing

    Note: Only use power-of-2 head counts for compatibility with TRT-LLM kernel
    """
    configs = [
        # Small: typical decode batch
        dict(batch_size=1, num_heads=32, head_dim=128, seq_len=128, max_kv_len=128),
        dict(batch_size=8, num_heads=32, head_dim=128, seq_len=256, max_kv_len=256),
        dict(batch_size=16, num_heads=32, head_dim=128, seq_len=512, max_kv_len=512),

        # Medium: typical prefill (changed from 40 to 32 heads for compatibility)
        dict(batch_size=4, num_heads=32, head_dim=128, seq_len=1024, max_kv_len=1024),
        dict(batch_size=8, num_heads=32, head_dim=128, seq_len=2048, max_kv_len=2048),

        # Large: long context
        dict(batch_size=2, num_heads=64, head_dim=128, seq_len=4096, max_kv_len=4096),
    ]

    for config in configs:
        config['dtype'] = torch.float8_e4m3fn
        yield config


if __name__ == "__main__":
    # Initialize CSV file with header
    output_dir = "/tmp/fp4_test"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/flashinfer_prefill.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("dtype,batch_size,num_heads,head_dim,seq_len,max_kv_len,total_tokens,t_us,tflops,gb_per_s\n")

    print("=" * 80)
    print("FlashInfer Prefill Kernel Benchmark")
    print("=" * 80)

    # Use simple configs for quick testing, or enumerate_test_configs() for comprehensive testing
    use_full_suite = os.environ.get("FULL_BENCH", "0") == "1"
    config_generator = enumerate_test_configs() if use_full_suite else enumerate_simple_configs()

    for config in config_generator:
        print(f"\nTesting config: {config}")
        try:
            bench_one(**config)
        except Exception as e:
            print(f"Error for config {config}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"Results saved to {csv_path}")
    print("=" * 80)

