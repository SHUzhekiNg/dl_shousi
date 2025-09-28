import numpy as np

def conv2d_numpy(input_array, kernel, stride=1, padding=0, dilation=1):
    """
    使用numpy实现2D卷积
    
    Args:
        input_array: 输入数组，shape为 (batch_size, in_channels, height, width)
        kernel: 卷积核，shape为 (out_channels, in_channels, kernel_height, kernel_width)
        stride: 步长，可以是int或tuple (stride_h, stride_w)
        padding: 填充，可以是int或tuple (pad_h, pad_w)
        dilation: 膨胀率，可以是int或tuple (dilation_h, dilation_w)
    
    Returns:
        输出数组，shape为 (batch_size, out_channels, out_height, out_width)
    """
    # 处理参数格式
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    # 获取输入和卷积核的维度
    batch_size, in_channels, input_h, input_w = input_array.shape
    out_channels, kernel_in_channels, kernel_h, kernel_w = kernel.shape
    
    assert in_channels == kernel_in_channels, "输入通道数与卷积核通道数不匹配"
    
    # 计算膨胀后的卷积核大小
    dilated_kernel_h = kernel_h + (kernel_h - 1) * (dilation[0] - 1)
    dilated_kernel_w = kernel_w + (kernel_w - 1) * (dilation[1] - 1)
    
    # 添加padding
    if padding[0] > 0 or padding[1] > 0:
        padded_input = np.pad(input_array, pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), mode='constant', constant_values=0)
    else:
        padded_input = input_array
    
    # 计算输出尺寸
    padded_h, padded_w = padded_input.shape[2], padded_input.shape[3]
    out_h = (padded_h - dilated_kernel_h) // stride[0] + 1
    out_w = (padded_w - dilated_kernel_w) // stride[1] + 1
    
    # 初始化输出
    output = np.zeros((batch_size, out_channels, out_h, out_w))
    
    # 执行卷积操作
    for b in range(batch_size):
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    # 计算当前输出位置对应的输入区域
                    h_start = oh * stride[0]
                    w_start = ow * stride[1]
                    
                    # 初始化当前输出值
                    conv_sum = 0
                    
                    # 对所有输入通道进行卷积
                    for ic in range(in_channels):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                # 计算膨胀卷积的实际位置
                                h_pos = h_start + kh * dilation[0]
                                w_pos = w_start + kw * dilation[1]
                                
                                # 检查边界
                                if (h_pos < padded_h and w_pos < padded_w):
                                    conv_sum += padded_input[b, ic, h_pos, w_pos] * kernel[oc, ic, kh, kw]
                    
                    output[b, oc, oh, ow] = conv_sum
    
    return output

def conv1d_numpy(input_array, kernel, stride=1, padding=0, dilation=1):
    """
    使用numpy实现1D卷积
    
    Args:
        input_array: 输入数组，shape为 (batch_size, in_channels, length)
        kernel: 卷积核，shape为 (out_channels, in_channels, kernel_size)
        stride: 步长
        padding: 填充
        dilation: 膨胀率
    
    Returns:
        输出数组，shape为 (batch_size, out_channels, out_length)
    """
    # 获取维度
    batch_size, in_channels, input_length = input_array.shape
    out_channels, kernel_in_channels, kernel_size = kernel.shape
    
    assert in_channels == kernel_in_channels, "输入通道数与卷积核通道数不匹配"
    
    # 计算膨胀后的卷积核大小
    dilated_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    
    # 添加padding
    if padding > 0:
        padded_input = np.pad(input_array, ((0, 0), (0, 0), (padding, padding)), mode='constant')
    else:
        padded_input = input_array
    
    # 计算输出长度
    padded_length = padded_input.shape[2]
    out_length = (padded_length - dilated_kernel_size) // stride + 1
    
    # 初始化输出
    output = np.zeros((batch_size, out_channels, out_length))
    
    # 执行卷积
    for b in range(batch_size):
        for oc in range(out_channels):
            for ol in range(out_length):
                l_start = ol * stride
                conv_sum = 0
                
                for ic in range(in_channels):
                    for k in range(kernel_size):
                        l_pos = l_start + k * dilation
                        if l_pos < padded_length:
                            conv_sum += padded_input[b, ic, l_pos] * kernel[oc, ic, k]
                
                output[b, oc, ol] = conv_sum
    
    return output

# 测试函数
def test_conv2d():
    """测试2D卷积实现"""
    print("=== 测试2D卷积 ===")
    
    # 创建测试数据
    np.random.seed(42)
    input_data = np.random.randn(2, 3, 8, 8)  # (batch_size=2, channels=3, height=8, width=8)
    kernel = np.random.randn(16, 3, 3, 3)     # (out_channels=16, in_channels=3, kernel_h=3, kernel_w=3)
    
    print(f"输入shape: {input_data.shape}")
    print(f"卷积核shape: {kernel.shape}")
    
    # 基本卷积
    output1 = conv2d_numpy(input_data, kernel, stride=1, padding=0)
    print(f"基本卷积(stride=1, padding=0)输出shape: {output1.shape}")
    
    # 带padding的卷积
    output2 = conv2d_numpy(input_data, kernel, stride=1, padding=1)
    print(f"padding=1卷积输出shape: {output2.shape}")
    
    # 带stride的卷积
    output3 = conv2d_numpy(input_data, kernel, stride=2, padding=1)
    print(f"stride=2卷积输出shape: {output3.shape}")
    
    # 膨胀卷积
    output4 = conv2d_numpy(input_data, kernel, stride=1, padding=2, dilation=2)
    print(f"dilation=2卷积输出shape: {output4.shape}")

def test_conv1d():
    """测试1D卷积实现"""
    print("\n=== 测试1D卷积 ===")
    
    # 创建测试数据
    np.random.seed(42)
    input_data = np.random.randn(2, 4, 20)  # (batch_size=2, channels=4, length=20)
    kernel = np.random.randn(8, 4, 5)       # (out_channels=8, in_channels=4, kernel_size=5)
    
    print(f"输入shape: {input_data.shape}")
    print(f"卷积核shape: {kernel.shape}")
    
    # 基本卷积
    output1 = conv1d_numpy(input_data, kernel, stride=1, padding=0)
    print(f"基本卷积(stride=1, padding=0)输出shape: {output1.shape}")
    
    # 带padding的卷积
    output2 = conv1d_numpy(input_data, kernel, stride=1, padding=2)
    print(f"padding=2卷积输出shape: {output2.shape}")
    
    # 带stride的卷积
    output3 = conv1d_numpy(input_data, kernel, stride=2, padding=2)
    print(f"stride=2卷积输出shape: {output3.shape}")

def test_simple_example():
    """简单的手工计算例子验证正确性"""
    print("\n=== 简单验证例子 ===")
    
    # 创建简单的测试数据
    input_data = np.array([[[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]]])  # (1, 1, 3, 3)
    
    kernel = np.array([[[[1, 0],
                        [0, 1]]]])  # (1, 1, 2, 2) 对角线卷积核
    
    print("输入:")
    print(input_data[0, 0])
    print("卷积核:")
    print(kernel[0, 0])
    
    # 执行卷积
    output = conv2d_numpy(input_data, kernel, stride=1, padding=0)
    print("输出:")
    print(output[0, 0])
    
    # 手工计算验证
    # 左上角: 1*1 + 5*1 = 6
    # 右上角: 2*1 + 6*1 = 8
    # 左下角: 4*1 + 8*1 = 12
    # 右下角: 5*1 + 9*1 = 14
    expected = np.array([[6, 8], [12, 14]])
    print("期望输出:")
    print(expected)
    print("验证通过:", np.allclose(output[0, 0], expected))

# 运行测试
if __name__ == "__main__":
    test_conv2d()
    test_conv1d()
    test_simple_example()