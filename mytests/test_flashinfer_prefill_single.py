"""
Quick test script for testing a single FlashInfer prefill configuration.
Useful for debugging or quick performance checks.

Fixed architecture: 96 heads × 128 head_dim
"""

import torch
from bench_flashinfer_prefill import bench_one

if __name__ == "__main__":
    print("Testing single configuration for FlashInfer Prefill")
    print("=" * 80)

    # Configuration parameters
    # Fixed architecture
    num_heads = 96
    head_dim = 128

    # Example 1: Full prefill (no existing cache)
    config_full = dict(
        batch_size=4,
        num_heads=num_heads,
        head_dim=head_dim,
        q_len=1024,       # New tokens to prefill
        kv_len=1024,      # Total KV cache length (same as q_len for full prefill)
        dtype=torch.float8_e4m3fn,
    )

    # Example 2: Partial prefill (with existing cache)
    config_partial = dict(
        batch_size=8,
        num_heads=num_heads,
        head_dim=head_dim,
        q_len=5,          # Only 5 new tokens to prefill
        kv_len=1024,      # But KV cache has 1024 tokens already
        dtype=torch.float8_e4m3fn,
    )

    # Choose which config to test (or test both)
    configs_to_test = [
        ("Full Prefill", config_full),
        ("Partial Prefill (1024 existing + 5 new)", config_partial),
    ]

    for name, config in configs_to_test:
        print(f"\n{name}:")
        print(f"Configuration: {config}")
        print()

        try:
            bench_one(**config)
            print(f"\n{name} completed successfully!")
        except Exception as e:
            print(f"\n{name} failed with error: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 80)

    print("=" * 80)
