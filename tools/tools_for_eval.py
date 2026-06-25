import numpy as np
import scipy.stats as stats

def add_noise_to_translation(matrix, absolute_mean=5):
    """
    在刚体变换矩阵的位移部分添加噪声，噪声模长的绝对值的平均数为 absolute_mean。
    
    :param matrix: 4x4 刚体变换矩阵
    :param absolute_mean: 噪声模长的绝对值的平均数，默认为 5
    :return: 添加噪声后的矩阵
    """
    # 确保矩阵是 4x4 的
    if matrix.shape != (4, 4):
        raise ValueError("输入矩阵必须是 4x4 的")
    
    # 计算所需的标准差（模长的期望值为 absolute_mean）
    noise_std = absolute_mean / np.sqrt(3)  # 因为模长是三个正态分布的分量的平方和

    # 提取平移部分（矩阵的最后一列）
    translation = matrix[:3, 3]
    
    # 生成三个方向的噪声分量
    noise = np.random.normal(0, noise_std, size=3)
    
    # 计算噪声的模长
    noise_magnitude = np.linalg.norm(noise)
    
    # 归一化噪声，使其模长为 absolute_mean
    noise_normalized = noise / noise_magnitude * absolute_mean
    
    # 添加噪声到平移部分
    new_translation = translation + noise_normalized
    
    # 将新的平移向量赋回矩阵
    matrix[:3, 3] = new_translation
    
    return matrix

def add_noise_to_translation_new(matrix, thres=5, percentile=70):
    """
    在刚体变换矩阵的位移部分添加噪声，噪声模长的绝对值的平均数为 absolute_mean。
    
    :param matrix: 4x4 刚体变换矩阵
    :param absolute_mean: 噪声模长的绝对值的平均数，默认为 5
    :return: 添加噪声后的矩阵
    """
    # 确保矩阵是 4x4 的
    if matrix.shape != (4, 4):
        raise ValueError("输入矩阵必须是 4x4 的")


    # 提取平移部分（矩阵的最后一列）
    translation = matrix[:3, 3]
    
    # 生成三个方向的噪声分量
    generate_noise()
    
    # 添加噪声到平移部分
    new_translation = translation + noise_normalized
    
    # 将新的平移向量赋回矩阵
    matrix[:3, 3] = new_translation
    
    return matrix

def generate_noise(size, percentile, mean=0, std=1):
    """
    生成符合正态分布的噪声，并确保前 percentile% 的数据落在 [-1,1] 之间。

    参数:
        size (int): 生成的噪声数量。
        percentile (float): 确保百分之多少的数据落在 [-1,1] 区间内（0-100）。
        mean (float): 正态分布的均值。
        std (float): 正态分布的标准差。

    返回:
        noise (numpy.ndarray): 生成的噪声数组。
    """
    # 计算正态分布的分位数，使前 percentile% 的数据落在 [-1,1] 内
    lower_bound = stats.norm.cdf(-1, loc=mean, scale=std)
    upper_bound = stats.norm.cdf(1, loc=mean, scale=std)
    
    # 计算当前分布中落在 [-1,1] 区间内的比例
    current_percent = (upper_bound - lower_bound) * 100
    
    # 调整标准差，使得前 percentile% 的数据落在 [-1,1] 内
    if current_percent < percentile:
        scale_factor = stats.norm.ppf(percentile / 100, loc=mean, scale=std) / 1
        std = std / scale_factor

    # 生成符合调整后标准差的噪声
    noise = np.random.normal(mean, std, size)
    
    return noise