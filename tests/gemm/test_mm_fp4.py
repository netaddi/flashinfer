import pytest
import torch
import torch.nn.functional as F
import gc
import itertools
import numpy as np
import pandas as pd

from flashinfer import (
    SfLayout,
    autotune,
    mm_fp4,
    nvfp4_quantize,
    mxfp4_quantize,
    testing,
)
from flashinfer.utils import get_compute_capability, LibraryError


def benchmark_fn(fn, docstring: str):
    measured_times = testing.bench_gpu_time_with_cuda_event(fn, dry_run_iters=5, repeat_iters=10)
    ms = np.average(measured_times)
    print("\n \033[1;32m" + docstring + f" Benchmark Time: {ms:.3f} ms" + "\033[0m")
    return ms


def save_test_results_to_excel(test_results, filename, test_name=None, show_stats=True, verbose=True):
    """
    将测试结果保存到Excel文件的通用函数

    Parameters:
    -----------
    test_results : list
        包含测试结果字典的列表，每个字典包含测试参数和结果
    filename : str
        Excel文件名，可以包含或不包含.xlsx扩展名
    test_name : str, optional
        测试名称，用于打印信息。如果为None，使用filename
    show_stats : bool, optional
        是否显示统计信息，默认True
    verbose : bool, optional
        是否显示详细进度信息，默认True

    Returns:
    --------
    dict
        包含保存结果的统计信息
    """
    if not test_results:
        if verbose:
            print("没有测试结果需要保存")
        return {"saved": False, "total_tests": 0, "successful_tests": 0, "failed_tests": 0}

    # 确保文件名有正确的扩展名
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'

    # 使用文件名作为默认测试名称
    if test_name is None:
        test_name = filename.replace('.xlsx', '')

    try:
        # 创建DataFrame
        df = pd.DataFrame(test_results)

        # 使用ExcelWriter来实现更好的格式控制
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='测试结果', index=False)

            # 获取工作表和工作簿对象
            worksheet = writer.sheets['测试结果']
            workbook = writer.book

            # 定义表头格式
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })

            # 定义数据格式
            cell_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'border': 1
            })

            # 应用表头格式
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # 计算每列的最佳宽度
            for i, col in enumerate(df.columns):
                # 计算列名长度
                column_len = len(str(col))

                # 计算该列数据的最大长度
                if len(df) > 0:
                    max_len = df[col].astype(str).str.len().max()
                    column_len = max(column_len, max_len)

                # 设置合理的列宽范围（最小8，最大50）
                column_len = min(max(column_len + 2, 8), 50)
                worksheet.set_column(i, i, column_len, cell_format)

            # 如果数据量大，启用自动筛选
            if len(df) > 1:
                worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)

        if verbose:
            print(f"\n测试结果已保存到 {filename}")
            print(f"总共完成 {len(test_results)} 个测试用例")

        # 统计信息
        successful_tests = df[df['latency_ms'].notna()]
        failed_tests_count = len(test_results) - len(successful_tests)

        stats = {
            "saved": True,
            "filename": filename,
            "total_tests": len(test_results),
            "successful_tests": len(successful_tests),
            "failed_tests": failed_tests_count
        }

        if show_stats and verbose:
            if len(successful_tests) > 0:
                print(f"成功测试: {len(successful_tests)} 个")
                if 'latency_ms' in successful_tests.columns:
                    stats["avg_latency"] = successful_tests['latency_ms'].mean()
                    stats["min_latency"] = successful_tests['latency_ms'].min()
                    stats["max_latency"] = successful_tests['latency_ms'].max()
                    print(f"平均延迟: {stats['avg_latency']:.3f} ms")
                    print(f"延迟范围: {stats['min_latency']:.3f} - {stats['max_latency']:.3f} ms")

            if failed_tests_count > 0:
                print(f"失败测试: {failed_tests_count} 个")

        return stats

    except Exception as e:
        error_msg = f"保存Excel文件时出错: {str(e)}"
        if verbose:
            print(error_msg)
        return {
            "saved": False,
            "error": error_msg,
            "total_tests": len(test_results),
            "successful_tests": 0,
            "failed_tests": len(test_results)
        }


def collect_test_result(test_params, latency=None, error=None, extra_data=None):
    """
    收集单个测试结果的辅助函数

    Parameters:
    -----------
    test_params : dict
        测试参数字典
    latency : float, optional
        测试延迟结果（毫秒）
    error : str, optional
        错误信息（如果测试失败）
    extra_data : dict, optional
        额外需要记录的数据

    Returns:
    --------
    dict
        格式化的测试结果字典
    """
    result = test_params.copy()

    if latency is not None:
        result['latency_ms'] = latency
    else:
        result['latency_ms'] = None

    if error is not None:
        result['error'] = error

    if extra_data:
        result.update(extra_data)

    return result


# TODO: Consdier splitting this function up for the various backends
@pytest.mark.parametrize("m", [1, 48, 128, 256, 512])
@pytest.mark.parametrize("n", [128, 256, 512])
@pytest.mark.parametrize("k", [128, 256, 512])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["trtllm", "cudnn", "cutlass"])
@pytest.mark.parametrize("use_128x4_sf_layout", [False, True])
@pytest.mark.parametrize("auto_tuning", [False, True])
@pytest.mark.parametrize("fp4_type", ["nvfp4", "mxfp4", "mxfp4_alpha"])
def test_mm_fp4(
    m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type
):
    use_nvfp4 = fp4_type == "nvfp4"

    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if backend == "trtllm":
        if res_dtype == torch.float16:
            pytest.skip("Skipping test for trtllm fp4 with float16")
        if compute_capability[0] in [11, 12]:
            pytest.skip("trtllm gemm does not support SM110/SM120/SM121 GPUs.")
    if not use_128x4_sf_layout and backend != "trtllm":
        pytest.skip("Skipping test for non-trtllm fp4 with use_128x4_sf_layout=False")
    if auto_tuning and backend == "cudnn":
        pytest.skip("Skipping test for cudnn fp4 with auto_tuning=True")
    if not use_nvfp4 and backend != "cudnn":
        pytest.skip("mx_fp4 is only supported for cudnn backend")

    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4

    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    # for trtllm, we need to shuffle mat2 because we swap A, B.
    do_shuffle_b = backend == "trtllm"

    block_size = 16 if use_nvfp4 else 32
    has_alpha = fp4_type == "mxfp4_alpha" or fp4_type == "nvfp4"

    if use_nvfp4:
        input_fp4, input_inv_s = nvfp4_quantize(
            input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
        )
        mat2_fp4, mat2_inv_s = nvfp4_quantize(
            mat2,
            global_sf_mat2,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=do_shuffle_b,
        )
    else:
        input_fp4, input_inv_s = mxfp4_quantize(input)
        mat2_fp4, mat2_inv_s = mxfp4_quantize(mat2)

    alpha = 1.0 / (global_sf_input * global_sf_mat2) if has_alpha else None

    reference = torch.mm(input, mat2.T)

    res = torch.empty([m, n], device="cuda", dtype=res_dtype)

    try:
        with autotune(auto_tuning):
            mm_fp4(
                input_fp4,
                mat2_fp4.T,
                input_inv_s,
                mat2_inv_s.T,
                alpha,
                res_dtype,
                res,
                block_size=block_size,
                use_8x4_sf_layout=not use_128x4_sf_layout,
                backend=backend,
                use_nvfp4=use_nvfp4,
            )

        cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
        assert cos_sim > 0.97

        # Benchmark kernel
        def fn():
            mm_fp4(
                input_fp4,
                mat2_fp4.T,
                input_inv_s,
                mat2_inv_s.T,
                alpha,
                res_dtype,
                res,
                block_size=block_size,
                use_8x4_sf_layout=not use_128x4_sf_layout,
                backend=backend,
                use_nvfp4=use_nvfp4,
            )

        docstring = f"test_mm_fp4(m={m}, n={n}, k={k}, res_dtype={res_dtype}, backend={backend}, use_128x4_sf_layout={use_128x4_sf_layout}, fp4_type={fp4_type})"
        latency = benchmark_fn(fn, docstring)
        return latency

    except LibraryError:
        # TODO: Remove this check once cuDNN backend version is updated to 9.14.0
        if (
            backend == "cudnn"
            and not use_nvfp4
            and (compute_capability[0] == 12 and compute_capability[1] == 0)
        ):
            pytest.xfail(
                "cudnn FP4 GEMM with mxfp4 quantization is not supported on SM120 with cuDNN backend version < 9.14.0."
            )
        else:
            pytest.fail("Unexpected LibraryError")


def benchmark_mm_fp4():
    """
    Benchmark function for mm_fp4 that iterates through parameter combinations
    """
    # Define parameter ranges
    m_values = [1, 48, 128, 256, 512, 1024, 2048]
    n_values = [128, 256, 512, 1024, 2048]
    k_values = [128, 256, 512, 1024, 2048, 4096]
    res_dtype_values = [torch.bfloat16, torch.float16]
    backend_values = ["trtllm", "cudnn", "cutlass"]
    use_128x4_sf_layout_values = [False, True]
    auto_tuning_values = [False, True]
    fp4_type_values = ["nvfp4", "mxfp4", "mxfp4_alpha"]

    test_results = []

    for m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type in itertools.product(
        m_values, n_values, k_values, res_dtype_values,
        backend_values, use_128x4_sf_layout_values, auto_tuning_values, fp4_type_values
    ):
        # Skip invalid combinations
        compute_capability = get_compute_capability(torch.device(device="cuda"))

        if backend == "trtllm":
            if res_dtype == torch.float16:
                continue
            if compute_capability[0] in [11, 12]:
                continue

        if not use_128x4_sf_layout and backend != "trtllm":
            continue

        if auto_tuning and backend == "cudnn":
            continue

        use_nvfp4 = fp4_type == "nvfp4"
        if not use_nvfp4 and backend != "cudnn":
            continue

        test_params = {
            'm': m,
            'n': n,
            'k': k,
            'res_dtype': str(res_dtype),
            'backend': backend,
            'use_128x4_sf_layout': use_128x4_sf_layout,
            'auto_tuning': auto_tuning,
            'fp4_type': fp4_type
        }

        try:
            latency = test_mm_fp4(m, n, k, res_dtype, backend, use_128x4_sf_layout, auto_tuning, fp4_type)
            result_data = collect_test_result(test_params, latency=latency)
            test_results.append(result_data)
            print(f"测试完成 - m={m}, n={n}, k={k}, backend={backend}, fp4_type={fp4_type}, latency={latency:.3f} ms")

        except Exception as e:
            error_msg = str(e)
            print(f"测试失败 - m={m}, n={n}, k={k}, backend={backend}, fp4_type={fp4_type}, error={error_msg}")
            result_data = collect_test_result(test_params, error=error_msg)
            test_results.append(result_data)

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # Save results to Excel
    save_test_results_to_excel(test_results, "test_mm_fp4_benchmark", show_stats=True)


if __name__ == "__main__":
    benchmark_mm_fp4()
