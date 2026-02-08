import numpy as np


def conv2d_numpy(input_array, kernel, stride=1, padding=0, dilation=1):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    batch_size, in_channels, input_h, input_w = input_array.shape
    out_channels, kernel_in_channels, kernel_h, kernel_w = kernel.shape
    assert in_channels == kernel_in_channels

    dilated_kh = kernel_h + (kernel_h - 1) * (dilation[0] - 1)
    dilated_kw = kernel_w + (kernel_w - 1) * (dilation[1] - 1)

    if padding[0] > 0 or padding[1] > 0:
        padded = np.pad(input_array, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))
    else:
        padded = input_array

    pad_h, pad_w = padded.shape[2], padded.shape[3]
    out_h = (pad_h - dilated_kh) // stride[0] + 1
    out_w = (pad_w - dilated_kw) // stride[1] + 1
    output = np.zeros((batch_size, out_channels, out_h, out_w))

    for b in range(batch_size):
        for oc in range(out_channels):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride[0]
                    w_start = ow * stride[1]
                    conv_sum = 0.0
                    for ic in range(in_channels):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                h_pos = h_start + kh * dilation[0]
                                w_pos = w_start + kw * dilation[1]
                                if h_pos < pad_h and w_pos < pad_w:
                                    conv_sum += padded[b, ic, h_pos, w_pos] * kernel[oc, ic, kh, kw]
                    output[b, oc, oh, ow] = conv_sum

    return output


def conv1d_numpy(input_array, kernel, stride=1, padding=0, dilation=1):
    batch_size, in_channels, input_length = input_array.shape
    out_channels, kernel_in_channels, kernel_size = kernel.shape
    assert in_channels == kernel_in_channels

    dilated_ks = kernel_size + (kernel_size - 1) * (dilation - 1)

    if padding > 0:
        padded = np.pad(input_array, ((0, 0), (0, 0), (padding, padding)))
    else:
        padded = input_array

    pad_len = padded.shape[2]
    out_len = (pad_len - dilated_ks) // stride + 1
    output = np.zeros((batch_size, out_channels, out_len))

    for b in range(batch_size):
        for oc in range(out_channels):
            for ol in range(out_len):
                l_start = ol * stride
                conv_sum = 0.0
                for ic in range(in_channels):
                    for k in range(kernel_size):
                        l_pos = l_start + k * dilation
                        if l_pos < pad_len:
                            conv_sum += padded[b, ic, l_pos] * kernel[oc, ic, k]
                output[b, oc, ol] = conv_sum

    return output


if __name__ == "__main__":
    np.random.seed(42)

    x2d = np.random.randn(2, 3, 8, 8)
    w2d = np.random.randn(16, 3, 3, 3)
    for s, p, d, tag in [(1, 0, 1, "basic"), (1, 1, 1, "pad=1"), (2, 1, 1, "stride=2"), (1, 2, 2, "dilation=2")]:
        out = conv2d_numpy(x2d, w2d, stride=s, padding=p, dilation=d)
        print(f"Conv2d {tag}: {x2d.shape} -> {out.shape}")

    x1d = np.random.randn(2, 4, 20)
    w1d = np.random.randn(8, 4, 5)
    for s, p, d, tag in [(1, 0, 1, "basic"), (1, 2, 1, "pad=2"), (2, 2, 1, "stride=2")]:
        out = conv1d_numpy(x1d, w1d, stride=s, padding=p, dilation=d)
        print(f"Conv1d {tag}: {x1d.shape} -> {out.shape}")

    x_simple = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=float)
    k_simple = np.array([[[[1, 0], [0, 1]]]], dtype=float)
    out_simple = conv2d_numpy(x_simple, k_simple)
    expected = np.array([[6, 8], [12, 14]])
    print(f"Simple verify: {np.allclose(out_simple[0, 0], expected)}")
