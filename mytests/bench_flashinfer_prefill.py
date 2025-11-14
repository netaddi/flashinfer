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
    q_len,
    kv_len,
    block_size=64,
    dtype=torch.float8_e4m3fn,
):
    """
    Benchmark trtllm_batch_context_with_kv_cache kernel

    Args:
        batch_size: Number of sequences in the batch
        num_heads: Number of attention heads (both Q and KV)
        head_dim: Dimension of each attention head
        q_len: Query sequence length (new tokens to prefill)
        kv_len: Total KV cache length (including existing cache)
        block_size: KV cache block size (page size)
        dtype: Data type for computation

    Note: Supports partial prefill where kv_len > q_len
          (e.g., kv_len=1024 existing cache, q_len=5 new tokens)
    """

    data = create_data(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        q_len=q_len,
        kv_len=kv_len,
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
        return o

    # First, test if the kernel can run successfully
    try:
        _ = test_func()
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  ! Kernel execution failed: {e}")
        raise

    # Benchmark using kineto
    t = bench_kineto(
        test_func,
        "fmha",
        suppress_kineto_output=True,
        num_tests=8,
    )

    # Calculate metrics
    total_q_tokens = batch_size * q_len  # Total query tokens

    # FLOPS calculation for attention:
    # BMM1: Q @ K^T = [total_q_tokens, num_heads, head_dim] @ [num_heads, head_dim, kv_len]
    #       -> [total_q_tokens, num_heads, kv_len]
    # BMM2: S @ V = [total_q_tokens, num_heads, kv_len] @ [num_heads, kv_len, head_dim]
    #       -> [total_q_tokens, num_heads, head_dim]
    flops_bmm1 = 2 * total_q_tokens * num_heads * kv_len * head_dim
    flops_bmm2 = 2 * total_q_tokens * num_heads * kv_len * head_dim
    total_flops = flops_bmm1 + flops_bmm2
    tflops = total_flops / t / 1e12

    # Memory bandwidth calculation
    # Input: Q (q_len tokens), K cache (kv_len tokens), V cache (kv_len tokens)
    # Output: O (q_len tokens)
    query_bytes = data["query"].numel() * data["query"].element_size()
    kv_bytes = (data["k_cache"].numel() + data["v_cache"].numel()) * data["k_cache"].element_size()
    output_bytes = total_q_tokens * num_heads * head_dim * torch.finfo(dtype).bits // 8
    total_bytes = query_bytes + kv_bytes + output_bytes
    gb_per_s = total_bytes / 1e9 / t

    print(
        f" > Perf (bs={batch_size:2}, nh={num_heads:3}, hd={head_dim:3}, q={q_len:4}, kv={kv_len:4}): "
        f"{t * 1e6:7.0f} us | {tflops:6.2f} TFLOPS | {gb_per_s:6.0f} GB/s"
    )

    # Save results to CSV
    output_dir = "/tmp/fp4_test"
    os.makedirs(output_dir, exist_ok=True)
    f = open(f"{output_dir}/flashinfer_prefill.csv", "a")
    f.write(
        f"{dtype},{batch_size},{num_heads},{head_dim},{q_len},{kv_len},"
        f"{total_q_tokens},{t*1e6},{tflops},{gb_per_s}\n"
    )
    f.close()

    metrics = dict(
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        q_len=q_len,
        kv_len=kv_len,
        total_q_tokens=total_q_tokens,
        t_us=t * 1e6,
        tflops=tflops,
        gb_per_s=gb_per_s,
    )
    print(f"MAIN_OUTPUT={json.dumps(metrics)}")


def create_data(
    batch_size,
    num_heads,
    head_dim,
    q_len,
    kv_len,
    block_size=64,
    dtype=torch.float8_e4m3fn,
    device="cuda:0",
):
    """
    Create test data for FlashInfer prefill kernel with partial prefill support

    Args:
        q_len: Number of new query tokens to prefill
        kv_len: Total KV cache length (can be > q_len for partial prefill)

    Returns a dictionary with:
        - query: [total_q_tokens, num_heads, head_dim]
        - k_cache, v_cache: KV cache blocks
        - workspace_buffer: workspace memory
        - block_tables: [batch_size, max_blocks]
        - seq_lens: [batch_size] - KV sequence lengths
        - cum_seq_lens_q, cum_seq_lens_kv: cumulative sequence lengths
        - bmm1_scale, bmm2_scale: scaling factors
        - max_q_len, max_kv_len: maximum lengths
    """
    device = torch.device(device)

    # Create query tensor: [total_q_tokens, num_heads, head_dim]
    # Note: torch.randn doesn't support FP8, so we create with bfloat16 then convert
    total_q_tokens = batch_size * q_len
    query = torch.randn(total_q_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device).to(dtype)

    # Query sequence lengths (all same for simplicity)
    q_seq_lens = torch.full((batch_size,), q_len, dtype=torch.int32, device=device)

    # KV sequence lengths (can be larger than q_seq_lens for partial prefill)
    kv_seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Calculate cumulative sequence lengths
    cum_seq_lens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cum_seq_lens_q[1:] = torch.cumsum(q_seq_lens, dim=0)

    cum_seq_lens_kv = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cum_seq_lens_kv[1:] = torch.cumsum(kv_seq_lens, dim=0)

    # Create KV cache blocks based on kv_len
    # Calculate number of blocks needed for KV cache
    num_blocks_per_seq = (kv_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # For MHA (Multi-Head Attention), num_kv_heads should equal num_heads
    # For GQA (Grouped Query Attention), num_kv_heads < num_heads
    # We use MHA here for simplicity
    num_kv_heads = num_heads

    # KV cache shape: [num_pages, num_kv_heads, page_size, head_dim]
    # Note: torch.randn doesn't support FP8, so we create with bfloat16 then convert
    k_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16, device=device).to(dtype)
    v_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16, device=device).to(dtype)

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
        seq_lens=kv_seq_lens,  # Kernel expects KV sequence lengths
        max_q_len=q_len,
        max_kv_len=kv_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        batch_size=batch_size,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
    )


def enumerate_test_configs():
    """
    Generate test configurations for fixed model architecture (96 heads × 128 head_dim)
    Tests various batch sizes and sequence lengths, including partial prefill scenarios

    Partial prefill: When kv_len > q_len, simulating continuous batching where
    existing KV cache is already filled, and we only prefill new tokens.

    Example: kv_len=1024 (existing cache), q_len=5 (new tokens to add)
    """
    # Fixed model architecture
    num_heads = 96
    head_dim = 128

    # Batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    # Full prefill scenarios: q_len == kv_len (no existing cache)
    full_prefill_lens = [128, 256, 512, 1024, 2048, 4096]

    # Partial prefill scenarios: (kv_len, q_len)
    # Simulating: existing cache + new tokens
    partial_prefill_configs = [
        (128, 5),     # 128 existing, 5 new
        (256, 5),     # 256 existing, 5 new
        (512, 5),     # 512 existing, 5 new
        (1024, 5),    # 1024 existing, 5 new
        (1024, 32),   # 1024 existing, 32 new
        (2048, 5),    # 2048 existing, 5 new
        (2048, 64),   # 2048 existing, 64 new
        (4096, 5),    # 4096 existing, 5 new
        (4096, 128),  # 4096 existing, 128 new
    ]

    # Full prefill scenarios
    for q_len in full_prefill_lens:
        kv_len = q_len  # Full prefill: no existing cache

        # Adjust batch sizes based on memory constraints
        if q_len >= 4096:
            valid_batch_sizes = [1, 2, 4, 8]
        elif q_len >= 2048:
            valid_batch_sizes = [1, 2, 4, 8, 16]
        elif q_len >= 1024:
            valid_batch_sizes = [1, 2, 4, 8, 16, 32]
        else:
            valid_batch_sizes = batch_sizes

        for batch_size in valid_batch_sizes:
            yield dict(
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                q_len=q_len,
                kv_len=kv_len,
                dtype=torch.float8_e4m3fn,
            )

    # Partial prefill scenarios
    for kv_len, q_len in partial_prefill_configs:
        # Memory constraint: mostly limited by kv_len
        if kv_len >= 4096:
            valid_batch_sizes = [1, 2, 4]
        elif kv_len >= 2048:
            valid_batch_sizes = [1, 2, 4, 8, 16]
        elif kv_len >= 1024:
            valid_batch_sizes = [1, 2, 4, 8, 16, 32]
        else:
            valid_batch_sizes = batch_sizes

        for batch_size in valid_batch_sizes:
            yield dict(
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                q_len=q_len,
                kv_len=kv_len,
                dtype=torch.float8_e4m3fn,
            )


def enumerate_simple_configs():
    """
    Generate a smaller set of test configurations for quick testing
    Fixed architecture: 96 heads × 128 head_dim

    Includes both full and partial prefill scenarios
    """
    num_heads = 96
    head_dim = 128

    configs = []

    # test for prefill
    test_batch_sizes = [1, 2, 4, 8, 12, 16]
    test_prefill_lens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    for batch_size in test_batch_sizes:
        for q_len in test_prefill_lens:
            kv_len = q_len
            configs.append(dict(
                batch_size=batch_size,
                num_heads=num_heads,
                head_dim=head_dim,
                q_len=q_len,
                kv_len=kv_len,
                dtype=torch.float8_e4m3fn,
            ))

    # test for sp decode
    test_batch_sizes = list(range(32, 1100, 32))
    test_kv_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    test_q_lens = [1, 2, 3, 4, 5]

    for batch_size in test_batch_sizes:
        for kv_len in test_kv_lens:
            for q_len in test_q_lens:
                configs.append(dict(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    q_len=q_len,
                    kv_len=kv_len,
                    dtype=torch.float8_e4m3fn,
                ))

    for config in configs:
        yield config


if __name__ == "__main__":
    # Initialize CSV file with header
    output_dir = "/tmp/fp4_test"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/flashinfer_prefill.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("dtype,batch_size,num_heads,head_dim,q_len,kv_len,total_q_tokens,t_us,tflops,gb_per_s\n")

    print("=" * 80)
    print("FlashInfer Prefill Kernel Benchmark")
    print("Fixed architecture: 96 heads × 128 head_dim")
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

