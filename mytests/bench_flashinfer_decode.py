"""
Benchmark for FlashInfer decode kernel: trtllm_batch_decode_with_kv_cache

This test benchmarks the TRT-LLM batch decode attention kernel used in FlashInfer.
It measures performance metrics (latency, TFLOPS, bandwidth) for various batch sizes,
sequence lengths, and model configurations.

Focus: Testing performance with q_len_per_req > 1 (speculative decoding scenarios)
"""

import json
import os
import torch

try:
    import flashinfer
    from flashinfer.testing.utils import bench_kineto, count_bytes
except ImportError:
    print("Skipping bench_flashinfer_decode.py since required modules are not available")
    exit(0)


def bench_one(
    batch_size,
    num_heads,
    num_kv_heads,
    head_dim,
    q_len_per_req,
    kv_len,
    block_size=64,
    dtype=torch.float8_e4m3fn,
):
    """
    Benchmark trtllm_batch_decode_with_kv_cache kernel

    Args:
        batch_size: Number of sequences in the batch
        num_heads: Number of query attention heads
        num_kv_heads: Number of key-value attention heads (for GQA)
        head_dim: Dimension of each attention head
        q_len_per_req: Query length per request (1 for normal decode, >1 for speculative decoding)
        kv_len: Total KV cache length (existing cache)
        block_size: KV cache block size (page size)
        dtype: Data type for computation

    Note: Supports GQA where num_kv_heads < num_heads
          Supports speculative decoding where q_len_per_req > 1
    """

    data = create_data(
        batch_size=batch_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_len_per_req=q_len_per_req,
        kv_len=kv_len,
        block_size=block_size,
        dtype=dtype,
    )

    def test_func():
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=data["query"],
            kv_cache=(data["k_cache"], data["v_cache"]),
            workspace_buffer=data["workspace_buffer"],
            block_tables=data["block_tables"],
            seq_lens=data["seq_lens"],
            max_seq_len=data["max_seq_len"],
            bmm1_scale=data["bmm1_scale"],
            bmm2_scale=data["bmm2_scale"],
            window_left=-1,
            out_dtype=dtype,
            enable_pdl=None,
            sinks=None,
            q_len_per_req=q_len_per_req,
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
    total_q_tokens = batch_size * q_len_per_req  # Total query tokens
    total_kv_tokens = batch_size * kv_len  # Total KV tokens

    # FLOPS calculation for GQA attention:
    # In decode with q_len_per_req, each request processes q_len_per_req query tokens
    # BMM1: Q @ K^T for each query head with its corresponding KV head
    #       Q: [total_q_tokens, num_heads, head_dim]
    #       K: [total_kv_tokens, num_kv_heads, head_dim] (broadcasted to match query heads)
    #       For each query token, it attends to kv_len KV tokens
    #       FLOPS: 2 * total_q_tokens * num_heads * kv_len * head_dim
    # BMM2: attention_weights @ V
    #       S: [total_q_tokens, num_heads, kv_len]
    #       V: [total_kv_tokens, num_kv_heads, head_dim] (broadcasted to match query heads)
    #       Output: [total_q_tokens, num_heads, head_dim]
    #       FLOPS: 2 * total_q_tokens * num_heads * kv_len * head_dim
    flops_bmm1 = 2 * total_q_tokens * num_heads * kv_len * head_dim
    flops_bmm2 = 2 * total_q_tokens * num_heads * kv_len * head_dim
    total_flops = flops_bmm1 + flops_bmm2
    tflops = total_flops / t / 1e12

    # Memory bandwidth calculation for GQA:
    # Input tensors read from memory:
    #   - Query: [total_q_tokens, num_heads, head_dim]
    #   - K cache: [num_blocks, num_kv_heads, block_size, head_dim]
    #   - V cache: [num_blocks, num_kv_heads, block_size, head_dim]
    # Output tensor written to memory:
    #   - Output: [total_q_tokens, num_heads, head_dim]
    query_bytes = total_q_tokens * num_heads * head_dim * torch.finfo(dtype).bits // 8
    k_cache_bytes = data["k_cache"].numel() * data["k_cache"].element_size()
    v_cache_bytes = data["v_cache"].numel() * data["v_cache"].element_size()
    kv_bytes = k_cache_bytes + v_cache_bytes
    output_bytes = total_q_tokens * num_heads * head_dim * torch.finfo(dtype).bits // 8
    total_bytes = query_bytes + kv_bytes + output_bytes
    gb_per_s = total_bytes / 1e9 / t

    group_size = num_heads // num_kv_heads
    print(
        f" > Perf (bs={batch_size:3}, nh={num_heads:3}, nkv={num_kv_heads:2}, gs={group_size:2}, "
        f"hd={head_dim:3}, q_len={q_len_per_req:2}, kv={kv_len:6}): "
        f"{t * 1e6:8.0f} us | {tflops:6.2f} TFLOPS | {gb_per_s:7.0f} GB/s"
    )

    # Save results to CSV
    output_dir = "/tmp/fp4_test"
    os.makedirs(output_dir, exist_ok=True)
    f = open(f"{output_dir}/flashinfer_decode.csv", "a")
    f.write(
        f"{dtype},{batch_size},{num_heads},{num_kv_heads},{group_size},{head_dim},{q_len_per_req},{kv_len},"
        f"{total_q_tokens},{t*1e6},{tflops},{gb_per_s}\n"
    )
    f.close()

    metrics = dict(
        batch_size=batch_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        head_dim=head_dim,
        q_len_per_req=q_len_per_req,
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
    num_kv_heads,
    head_dim,
    q_len_per_req,
    kv_len,
    block_size=64,
    dtype=torch.float8_e4m3fn,
    device="cuda:0",
):
    """
    Create test data for FlashInfer decode kernel with GQA and speculative decoding support

    Args:
        batch_size: Number of sequences in the batch
        num_heads: Number of query attention heads
        num_kv_heads: Number of key-value attention heads (for GQA)
        head_dim: Dimension of each attention head
        q_len_per_req: Number of query tokens per request (1 for normal decode, >1 for speculative)
        kv_len: Total KV cache length
        block_size: KV cache block size (page size)
        dtype: Data type for computation
        device: Device to create tensors on

    Returns a dictionary with:
        - query: [total_q_tokens, num_heads, head_dim] where total_q_tokens = batch_size * q_len_per_req
        - k_cache, v_cache: [num_pages, num_kv_heads, page_size, head_dim] KV cache blocks
        - workspace_buffer: workspace memory
        - block_tables: [batch_size, max_blocks]
        - seq_lens: [batch_size] - KV sequence lengths
        - bmm1_scale, bmm2_scale: scaling factors
        - max_seq_len: maximum sequence length
    """
    device = torch.device(device)

    # Create query tensor: [total_q_tokens, num_heads, head_dim]
    # total_q_tokens = batch_size * q_len_per_req
    # Note: torch.randn doesn't support FP8, so we create with bfloat16 then convert
    total_q_tokens = batch_size * q_len_per_req
    query = torch.randn(total_q_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device).to(dtype)

    # KV sequence lengths (all same for simplicity)
    kv_seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Create KV cache blocks based on kv_len
    # Calculate number of blocks needed for KV cache
    num_blocks_per_seq = (kv_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # KV cache shape: [num_pages, num_kv_heads, page_size, head_dim]
    # Note: torch.randn doesn't support FP8, so we create with bfloat16 then convert
    # For MHA: num_kv_heads == num_heads
    # For GQA: num_kv_heads < num_heads (e.g., group_size=12 means num_kv_heads = num_heads/12)
    k_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16, device=device).to(dtype)
    v_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16, device=device).to(dtype)

    # Create block tables: [batch_size, max_blocks_per_seq]
    block_tables = torch.arange(total_blocks, dtype=torch.int32, device=device).reshape(
        batch_size, num_blocks_per_seq
    )

    # Create workspace buffer (512 MB) - must be initialized to 0
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
        seq_lens=kv_seq_lens,
        max_seq_len=kv_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )


def enumerate_test_configs():
    """
    Generate test configurations for fixed model architecture (96 heads × 128 head_dim)
    with GQA (group_size=12, num_kv_heads=8)

    Focus on testing various q_len_per_req values (especially > 1) for speculative decoding
    """
    # Fixed model architecture with GQA
    num_heads = 96
    group_size = 12
    num_kv_heads = num_heads // group_size  # 96 / 12 = 8
    head_dim = 128

    # Batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    # KV cache lengths to test
    kv_lens = [512, 1024, 2048, 4096, 8192, 16384]

    # Query lengths per request (key feature: test q_len > 1 for speculative decoding)
    q_len_per_reqs = [1, 2, 3, 4, 5, 8, 16, 32]

    for kv_len in kv_lens:
        # Adjust batch sizes based on memory constraints
        if kv_len >= 16384:
            valid_batch_sizes = [1, 2, 4, 8, 16]
        elif kv_len >= 8192:
            valid_batch_sizes = [1, 2, 4, 8, 16, 32]
        elif kv_len >= 4096:
            valid_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        else:
            valid_batch_sizes = batch_sizes

        for batch_size in valid_batch_sizes:
            for q_len_per_req in q_len_per_reqs:
                # Skip very large configurations
                if batch_size * q_len_per_req > 2048:
                    continue

                yield dict(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_len_per_req=q_len_per_req,
                    kv_len=kv_len,
                    dtype=torch.float8_e4m3fn,
                )


def enumerate_simple_configs():
    """
    Generate a smaller set of test configurations for quick testing
    Fixed architecture: 96 heads × 128 head_dim with GQA (group_size=12, num_kv_heads=8)

    Focus on speculative decoding scenarios (q_len_per_req > 1)
    """
    num_heads = 96
    group_size = 12
    num_kv_heads = num_heads // group_size  # 96 / 12 = 8
    head_dim = 128

    configs = []

    test_batch_sizes_spec = list(range(16, 530, 16))  # 16, 32, 48, ..., 512
    test_kv_lens_spec = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    test_q_lens_spec = [1, 2, 3, 4, 5, 6, 7, 8]

    for batch_size in test_batch_sizes_spec:
        for kv_len in test_kv_lens_spec:
            for q_len_per_req in test_q_lens_spec:
                configs.append(dict(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_len_per_req=q_len_per_req,
                    kv_len=kv_len,
                    dtype=torch.float8_e4m3fn,
                ))

    for config in configs:
        yield config


if __name__ == "__main__":
    # Initialize CSV file with header
    output_dir = "/tmp/fp4_test"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/flashinfer_decode.csv"
    with open(csv_path, "w") as f:
        f.write("dtype,batch_size,num_heads,num_kv_heads,group_size,head_dim,q_len_per_req,kv_len,total_q_tokens,t_us,tflops,gb_per_s\n")

    print("=" * 80)
    print("FlashInfer Decode Kernel Benchmark (trtllm_batch_decode_with_kv_cache)")
    print("Fixed architecture: 96 heads × 128 head_dim with GQA (group_size=12, num_kv_heads=8)")
    print("Focus: Testing q_len_per_req > 1 for speculative decoding scenarios")
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
