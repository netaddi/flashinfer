"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time, count_bytes

CSV_PATH = "/tmp/fp4_test/grouped_gemm.csv"


def bench_groupwise_grouped_gemm_fp8_blackwell(
    batch_size, m, n, k, in_dtype, out_dtype
):
    torch.random.manual_seed(0)
    a = torch.randn(batch_size * m, k, device="cuda:0").to(in_dtype)
    b = torch.randn(batch_size, n, k, device="cuda:0").to(in_dtype)
    out = torch.empty(batch_size * m, n, device="cuda:0", dtype=out_dtype)

    a_scale = torch.randn(
        (k // 128, batch_size * m), dtype=torch.float32, device="cuda:0"
    )
    b_scale = torch.randn(
        (batch_size, k // 128, n // 128), dtype=torch.float32, device="cuda:0"
    )

    segment_offsets = torch.arange(
        0, (batch_size + 1) * m, m, device="cuda:0", dtype=torch.int32
    )

    measurements = bench_gpu_time(
        lambda: flashinfer.gemm.group_gemm_fp8_nt_groupwise(
            a, b, a_scale, b_scale, segment_offsets, out=out, mma_sm=2
        ),
        dry_run_time_ms=2,
        repeat_time_ms=32,
    )
    ms = np.median(measurements)
    t = ms / 1e3  # convert to seconds
    tflops = 2 * batch_size * m * n * k / t / 1e12
    gb_per_s = count_bytes(a, b, a_scale, b_scale, out) / 1e9 / t
    print(
        f"group_gemm_fp8_nt_groupwise batch_size={batch_size} m={m} n={n} k={k} in_dtype={in_dtype} out_dtype={out_dtype}: {tflops:.2f} TFLOPs/s | {gb_per_s:.2f} GB/s"
    )

    f = open(CSV_PATH, "a")
    f.write(
        f"{in_dtype},{out_dtype},group_gemm_fp8_nt_groupwise,{batch_size},{m},{n},{k},{ms},{tflops},{gb_per_s}\n"
    )


if __name__ == "__main__":
    # Clear CSV and write header
    with open(CSV_PATH, "w") as f:
        f.write("in_dtype,out_dtype,kernel,batch_size,m,n,k,ms,tflops,gb_per_s\n")

    # for batch_size in [1, 2, 4]:
    #     for m in [1, 16, 24]:
    #         for n in [1024, 2048, 4096, 8192]:
    #             for k in [1024, 2048, 4096, 8192]:
    #                 bench_groupwise_grouped_gemm_fp8_blackwell(
    #                     batch_size, m, n, k, torch.float8_e5m2, torch.bfloat16
    #                 )

    for batch_size in [1]:
        for m in [8, 12, 16, 24, 32, 36, 48, 64, 96, 128]:
            qkv_size = 14336
            o_size = 12288
            hidden_size = 6144
            nk_tuple_list = []
            for tp_size in [1, 2, 4, 8]:
                nk_tuple_list.append((qkv_size // tp_size, hidden_size))
                nk_tuple_list.append((hidden_size, o_size // tp_size))
            for (n, k) in nk_tuple_list:
                bench_groupwise_grouped_gemm_fp8_blackwell(
                    batch_size, m, n, k, torch.float8_e5m2, torch.bfloat16
                )


