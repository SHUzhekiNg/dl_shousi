import torch
import math


def flash_attention(Q, K, V, block_size=32):
    B, H, N, D = Q.shape
    scale = 1.0 / math.sqrt(D)
    O = torch.zeros_like(V)
    L = torch.zeros(B, H, N, 1, device=Q.device)
    M = torch.full((B, H, N, 1), float('-inf'), device=Q.device)

    num_blocks_kv = math.ceil(N / block_size)
    num_blocks_q = math.ceil(N / block_size)

    for j in range(num_blocks_kv):
        kv_start = j * block_size
        kv_end = min(kv_start + block_size, N)
        Kj = K[:, :, kv_start:kv_end, :]
        Vj = V[:, :, kv_start:kv_end, :]

        for i in range(num_blocks_q):
            q_start = i * block_size
            q_end = min(q_start + block_size, N)
            Qi = Q[:, :, q_start:q_end, :]

            S = (Qi @ Kj.transpose(-2, -1)) * scale

            M_old = M[:, :, q_start:q_end, :]
            M_new = torch.max(M_old, S.max(dim=-1, keepdim=True).values)

            P = torch.exp(S - M_new)
            L_old = L[:, :, q_start:q_end, :]
            L_new = torch.exp(M_old - M_new) * L_old + P.sum(dim=-1, keepdim=True)

            O[:, :, q_start:q_end, :] = (
                torch.exp(M_old - M_new) * L_old * O[:, :, q_start:q_end, :] + P @ Vj
            ) / L_new

            M[:, :, q_start:q_end, :] = M_new
            L[:, :, q_start:q_end, :] = L_new

    return O


def standard_attention(Q, K, V):
    scale = 1.0 / math.sqrt(Q.shape[-1])
    attn = torch.softmax(Q @ K.transpose(-2, -1) * scale, dim=-1)
    return attn @ V


if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, N, D = 2, 4, 64, 32
    Q = torch.randn(B, H, N, D)
    K = torch.randn(B, H, N, D)
    V = torch.randn(B, H, N, D)

    out_flash = flash_attention(Q, K, V, block_size=16)
    out_standard = standard_attention(Q, K, V)
    print("Max diff:", (out_flash - out_standard).abs().max().item())
    print("Flash output shape:", out_flash.shape)
