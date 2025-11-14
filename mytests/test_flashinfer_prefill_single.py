"""
Quick test script for testing a single FlashInfer prefill configuration.
Useful for debugging or quick performance checks.
"""

import torch
from bench_flashinfer_prefill import bench_one

# Example: Test a single configuration
# Typical Llama-2 7B configuration with batch_size=8, seq_len=512

if __name__ == "__main__":
    print("Testing single configuration for FlashInfer Prefill")
    print("=" * 80)

    # Configuration parameters
    config = dict(
        batch_size=8,          # Number of sequences in batch
        num_heads=32,          # Number of attention heads (Llama-2 style)
        head_dim=128,          # Head dimension
        seq_len=512,           # Sequence length
        max_kv_len=512,        # Maximum KV cache length
        block_size=64,         # KV cache block size
        dtype=torch.float8_e4m3fn,  # FP8 data type
    )

    print(f"Configuration: {config}")
    print()

    try:
        bench_one(**config)
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)

