import csv
import numpy as np
from scipy.optimize import bisect


def inverse_calculation(coeffs: list, data_path: str, threshold: float = 5e-4) -> dict:
    """单次处理数据的高精度验证函数"""
    results = {
        'total': 0,
        'within_threshold': 0,
        'max_error': 0.0,
        'avg_error': 0.0,
        'failed_cases': 0,
        'max_attempt_log': [],
        'details': []
    }

    def poly_func(x):
        """多项式计算函数"""
        return sum(c * (x  ** i) for i, c in enumerate(coeffs))

    def find_root(y_target, nonce, max_attempts=30):
        """增强型二分法求解器"""
        base_min, base_max = 1, 3
        x_low = nonce + base_min
        x_high = nonce + base_max
        step_multiplier = 1.6
        current_step = 0.001

        for attempt in range(max_attempts):
            try:
                return bisect(lambda x: poly_func(x) - y_target, x_low, x_high, xtol=1e-12)
            except ValueError:
                f_low = poly_func(x_low) - y_target
                f_high = poly_func(x_high) - y_target

                # 记录区间扩展日志
                results['max_attempt_log'].append({
                    'attempt': attempt,
                    'x_low': x_low,
                    'x_high': x_high,
                    'f_low': f_low,
                    'f_high': f_high
                })

                # 智能区间扩展
                if f_low * f_high > 0:
                    x_low -= current_step
                    x_high += current_step
                else:
                    if abs(f_low) < abs(f_high):
                        x_low -= current_step
                    else:
                        x_high += current_step

                current_step *= step_multiplier

        # 最终尝试放宽精度
        try:
            return bisect(lambda x: poly_func(x) - y_target, x_low, x_high, xtol=1e-8)
        except Exception as e:
            raise RuntimeError(f"求解失败: {str(e)}")

    # 单次数据处理流程
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        next(reader)  # 跳过说明行

        for row in reader:
            x_base_true = float(row[0])
            nonce = float(row[1])
            y = float(row[2])

            record = {
                'y': y,
                'actual_x': x_base_true,
                'predicted_x': None,
                'success': False,
                'error': None
            }

            try:
                # 执行逆计算
                x_total = find_root(y, nonce)
                x_base_pred = x_total - nonce
                error = abs(x_base_pred - x_base_true)

                # 更新统计信息
                results['total'] += 1
                results['max_error'] = max(results['max_error'], error)
                results['avg_error'] += error
                if error <= threshold:
                    results['within_threshold'] += 1

                # 更新记录信息
                record.update({
                    'predicted_x': x_base_pred,
                    'success': error <= threshold,
                    'error': error
                })

            except Exception as e:
                results['failed_cases'] += 1
                record['error'] = str(e)

            results['details'].append(record)

    # 后处理计算平均误差
    if results['total'] > 0:
        results['avg_error'] /= results['total']

    # 生成详细报告
    with open('validation_details.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['y_value', 'actual_x', 'predicted_x', 'is_success', 'error'])

        for record in results['details']:
            # 格式化错误信息
            error_str = (
                f"{record['error']:.8e}"
                if isinstance(record['error'], float)
                else record['error']
            )

            # 处理预测值为None的情况
            pred_x = (
                f"{record['predicted_x']:.8f}"
                if record['predicted_x'] is not None
                else ''
            )

            writer.writerow([
                f"{record['y']:.10f}",
                f"{record['actual_x']:.8f}",
                pred_x,
                'Yes' if record['success'] else 'No',
                error_str
            ])

    return results


if __name__ == "__main__":
    # 配置参数
    DATA_FILE = "polynomial_data.csv"
    COEFFS = np.loadtxt('coeff_8th.csv').tolist()
    np.set_printoptions(suppress=True)  # 禁用科学计数法显示

    # 执行验证
    results = inverse_calculation(COEFFS, DATA_FILE)

    # 输出增强版结果报告
    total_samples = results['total'] + results['failed_cases']
    print(f"总样本数: {total_samples}")
    print(f"成功计算: {results['total']} | 失败案例: {results['failed_cases']}")
    print(f"阈值命中率: {results['within_threshold'] / results['total']:.4%}")
    print(f"最大绝对误差: {results['max_error']:.6e}")
    print(f"平均绝对误差: {results['avg_error']:.6e}")
    print(f"详细报告已保存至: validation_details.csv")