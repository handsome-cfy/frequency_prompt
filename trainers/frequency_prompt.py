import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyPrompt(nn.Module):
    def __init__(self,num_bands, num_channels, num_prompts):
        super(FrequencyPrompt, self).__init__()
        # 初始化频率提示，每个通道有一个对应的提示
        self.prompt = nn.Parameter(torch.randn(num_bands, num_prompts, num_channels))

    def forward(self, featuremap):
        # 提取 frequency_bands，假设其大小为 [32, 8]
        frequency_bands= FeatureMap2FrequencyBands(featuremap)

        masked_frequency_bands = []
        # 访问不同频率的信息
        for index in range(len(frequency_bands[0])):

            # 对于每一个通道
            frequency_pixel = []
            for channel in range(len(frequency_bands)):
                frequency_pixel.append(torch.from_numpy(frequency_bands[channel][index]))
            # [32,HW]
            frequency_pixel = torch.stack(frequency_pixel)

            # 第一个不是频率信息，是相似性故跳过
            if index == 0:
                masked_frequency_bands.append(frequency_pixel)
                continue

            # 对应这个频率的prompt[channel=32,num_prompt=32]
            band_prompt = self.prompt[index]
            M = torch.matmul(band_prompt, frequency_pixel)

            mask = torch.sigmoid(M)

            masked_frequency_pixel = mask * frequency_pixel
            masked_frequency_bands.append(masked_frequency_pixel)

        # masked_frequency_bands = np.stack(masked_frequency_bands)
        masked_featuremap = FrequencyBands2FeatureMap(masked_frequency_bands)

        if featuremap.requires_grad:
            masked_featuremap.grad = featuremap.grad.clone
            masked_featuremap.requires_grad = True

        return masked_featuremap

def FeatureMap2FrequencyBands(featuremap,wavelet='db1',level=3):
    # 用于存放DWT结果的列表

    dwt_results = []
    # 对每个信号样本执行3级的DWT
    for signal in featuremap:
        # 将PyTorch张量转换为NumPy数组
        signal_np = signal.cpu().numpy()
        # 执行DWT
        coeffs = pywt.wavedec(signal_np, wavelet, level)
        # 将DWT结果添加到列表中
        test = pywt.waverec(coeffs, wavelet)
        dwt_results.append(coeffs)

    return dwt_results

def FrequencyBands2FeatureMap(FrequencyBands,wavelet='db1',level=3):
    a = len(FrequencyBands[0])
    dwt_results = []
    for channel in range(len(FrequencyBands[0])):
        channel_frequencies = []
        for signal in FrequencyBands:
            channel_frequencies.append(signal[channel].detach().numpy())

        reconstructed_signal = pywt.waverec(channel_frequencies, wavelet)
        dwt_results.append(torch.from_numpy(reconstructed_signal))
    return torch.stack(dwt_results)

# # 假设我们有 64 个通道和 2 个提示
# num_channels = 64
# num_prompts = 2
#
# # 创建 Frequency Prompt 模块
# fp = FrequencyPrompt(num_channels, num_prompts)
#
# # 假设我们有一些频率带数据，形状为 (batch_size, num_channels, height, width)
# # 这里使用随机数据模拟
# batch_size, num_channels, height, width = 10, 64, 32, 32
# frequency_bands = torch.randn(batch_size, num_channels, height, width)
#
# # 通过 Frequency Prompt 模块
# masked_bands = fp(frequency_bands)

if __name__ == '__main__':
    import torch
    import pywt
    import numpy as np

    # 假设 input_signals 是形状为 [32, 512] 的 PyTorch 张量
    input_signals = torch.randn(32, 512)

    # 选择一个小波函数，例如 'db1' 使用Daubechies小波
    wavelet = 'db1'

    # 设置变换的级别为3
    level = 3

    # 用于存放DWT结果的列表
    dwt_HH_results = []

    # 对每个信号样本执行3级的DWT
    for signal in input_signals:
        # 将PyTorch张量转换为NumPy数组
        signal_np = signal.numpy()
        # 执行DWT
        coeffs = pywt.wavedec(signal_np, wavelet, level)
        # 将DWT结果添加到列表中
        dwt_HH_results.append(coeffs)

    for _singal in dwt_HH_results:
        org = pywt.waverec(_singal, wavelet)


    fp = FrequencyPrompt(10,32,32)
    ans = fp(input_signals)
    print(1)