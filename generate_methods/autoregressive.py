import torch
import torch.nn as nn

class SimpleARModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, h=None):
        # x: [batch, seq_len]
        embeds = self.embedding(x) # [batch, seq_len, embed_dim]
        out, h = self.rnn(embeds, h) # out: [batch, seq_len, hidden_dim]
        logits = self.fc(out) # [batch, seq_len, vocab_size]
        return logits, h

    @torch.no_grad()
    def generate(self, start_token, max_len=20):
        """逐个 token 生成序列"""
        generated = [start_token]
        input_idx = torch.tensor([[start_token]])
        h = None
        
        for _ in range(max_len):
            logits, h = self.forward(input_idx, h)
            # 取最后一个时间步的输出，并选择概率最大的词
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_token)
            input_idx = torch.tensor([[next_token]])
            
            if next_token == 0: # 假设 0 是停止符
                break
        return generated

# 示例用法
vocab_size = 10
model = SimpleARModel(vocab_size, 16, 32)
# 模拟生成
sequence = model.generate(start_token=1)
print("生成的序列:", sequence)