import pytest
import torch
import torch.nn.functional as F
import gc
import itertools
import numpy as np
import pandas as pd

from flashinfer import autotune, bmm_fp8, testing
from flashinfer.utils import get_compute_capability


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


def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


@pytest.mark.parametrize("b", [1, 16])
@pytest.mark.parametrize("m", [48, 128])
@pytest.mark.parametrize("n", [80, 64])
@pytest.mark.parametrize("k", [64, 256])
@pytest.mark.parametrize("input_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("mat2_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["cudnn", "cublas", "cutlass", "auto"])
@pytest.mark.parametrize("auto_tuning", [True, False])
def test_bmm_fp8(b, m, n, k, input_dtype, mat2_dtype, res_dtype, backend, auto_tuning):
    if get_compute_capability(torch.device("cuda"))[0] == 12 and backend in [
        "cutlass",
        "auto",
    ]:
        # TODO(yongwwww): enable all test cases for SM120/121 CUTLASS bmm_fp8 backend
        pytest.xfail(
            "Not all test cases for CUTLASS bmm_fp8 on SM120/121 are passing at this moment"
        )
    if input_dtype == torch.float8_e5m2 and mat2_dtype == torch.float8_e5m2:
        pytest.skip("Invalid combination: both input and mat2 are e5m2")
    if input_dtype == torch.float8_e5m2 or mat2_dtype == torch.float8_e5m2:
        if backend == "cutlass":
            pytest.skip("Invalid combination: cutlass does not support e5m2")
    if auto_tuning and backend != "cutlass":
        pytest.skip("Invalid combination: auto_tuning only supported for cutlass")

    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    # mat2 row  major -> column major
    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)
    reference = torch.bmm(input, mat2)

    res = torch.empty([b, m, n], device="cuda", dtype=res_dtype)

    with autotune(auto_tuning):
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            res_dtype,
            res,
            backend=backend,
        )

    cos_sim = F.cosine_similarity(reference.reshape(-1), res.reshape(-1), dim=0)
    assert cos_sim > 0.99

    # Benchmark kernel
    def fn():
        bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            res_dtype,
            res,
            backend=backend,
        )

    docstring = f"test_bmm_fp8(b={b}, m={m}, n={n}, k={k}, input_dtype={input_dtype}, mat2_dtype={mat2_dtype}, res_dtype={res_dtype}, backend={backend})"
    latency = benchmark_fn(fn, docstring)
    return latency


def benchmark_bmm_fp8():
    """
    Benchmark function for bmm_fp8 that iterates through parameter combinations
    """
    # Define parameter ranges
    b_values = [1, 4, 16, 64, 128]
    m_values = [48, 128, 256, 512]
    n_values = [64, 128, 256, 512]
    k_values = [64, 128, 256, 512, 1024]
    input_dtype_values = [torch.float8_e4m3fn, torch.float8_e5m2]
    mat2_dtype_values = [torch.float8_e4m3fn, torch.float8_e5m2]
    res_dtype_values = [torch.bfloat16, torch.float16]
    backend_values = ["cudnn", "cublas", "cutlass", "auto"]
    auto_tuning_values = [False, True]

    test_results = []

    for b, m, n, k, input_dtype, mat2_dtype, res_dtype, backend, auto_tuning in itertools.product(
        b_values, m_values, n_values, k_values, input_dtype_values,
        mat2_dtype_values, res_dtype_values, backend_values, auto_tuning_values
    ):
        # Skip invalid combinations
        if input_dtype == torch.float8_e5m2 and mat2_dtype == torch.float8_e5m2:
            continue
        if (input_dtype == torch.float8_e5m2 or mat2_dtype == torch.float8_e5m2) and backend == "cutlass":
            continue
        if auto_tuning and backend != "cutlass":
            continue

        # Skip SM120/121 CUTLASS issues
        if get_compute_capability(torch.device("cuda"))[0] == 12 and backend in ["cutlass", "auto"]:
            continue

        test_params = {
            'b': b,
            'm': m,
            'n': n,
            'k': k,
            'input_dtype': str(input_dtype),
            'mat2_dtype': str(mat2_dtype),
            'res_dtype': str(res_dtype),
            'backend': backend,
            'auto_tuning': auto_tuning
        }

        try:
            latency = test_bmm_fp8(b, m, n, k, input_dtype, mat2_dtype, res_dtype, backend, auto_tuning)
            result_data = collect_test_result(test_params, latency=latency)
            test_results.append(result_data)
            print(f"测试完成 - b={b}, m={m}, n={n}, k={k}, backend={backend}, latency={latency:.3f} ms")

        except Exception as e:
            error_msg = str(e)
            print(f"测试失败 - b={b}, m={m}, n={n}, k={k}, backend={backend}, error={error_msg}")
            result_data = collect_test_result(test_params, error=error_msg)
            test_results.append(result_data)

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # Save results to Excel
    save_test_results_to_excel(test_results, "test_bmm_fp8_benchmark", show_stats=True)


if __name__ == "__main__":
    benchmark_bmm_fp8()
