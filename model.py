import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class ModelArgs:
    rnn_units: int = 128
    rnn_layers: int = 1
    head_dim: int = 256
    top_classifier_units: int = 256
    dim: int = 128
    ffn_hidden_dim: int = 512
    max_seq_len: int = 512
    dropout_p: float = 0.05
    norm_eps: float = 1e-5
    bias: bool = False
    device: str = 'cuda'

class CreditsRNN(nn.Module):
    def __init__(
        self,
        features,
        embedding_projections,
        args: ModelArgs
    ):
        super(CreditsRNN, self).__init__()
        self.credits_cat_embeddings = nn.ModuleList([self.create_embedding_projection(*embedding_projections[feature])
                                                      for feature in features])
        
        self.dropout2d = nn.Dropout2d(args.dropout_p)

        self.lstm = nn.LSTM(
            input_size=sum([embedding_projections[x][1] for x in features]),
            hidden_size=args.rnn_units,
            num_layers=args.rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Bidirectional LSTM doubles the hidden size
        self.hidden_size = args.rnn_units * 2

        self.attention_forward = Encoder(args)
        self.attention_backward = Encoder(args)

        self.dropout1d = nn.Dropout(args.dropout_p)
        
        self.head = nn.Sequential(
            nn.Linear(in_features=args.dim * 2, out_features=args.top_classifier_units),
            nn.ReLU(),
            nn.Linear(in_features=args.top_classifier_units, out_features=1)
        )

    def forward(self, features):
        embeddings = [embedding(features[i]) for i, embedding in enumerate(self.credits_cat_embeddings)]
        concated_embeddings = self.dropout2d(torch.cat(embeddings, dim=-1))

        lstm_output, (last_hidden, c_n) = self.lstm(concated_embeddings)

        lstm_output_forward = lstm_output[..., self.hidden_size//2:]
        lstm_output_backward = lstm_output[..., :self.hidden_size//2]

        attention_output_forward = self.attention_forward(lstm_output_forward)
        attention_output_backward = self.attention_backward(lstm_output_backward)
        attention_output = torch.cat([attention_output_forward, attention_output_backward], dim=-1)

        attention_output = self.dropout1d(attention_output)

        raw_output = self.head(attention_output)

        return raw_output

    @classmethod
    def create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(num_embeddings=cardinality + add_missing, embedding_dim=embed_size, padding_idx=padding_idx)

class RoPE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.freqs_complex = self.precompute_theta_pos_frequencies(args.head_dim, args.max_seq_len, device=args.device)

    def precompute_theta_pos_frequencies(self, head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
        # As written in the paragraph 3.2.2 of the paper
        # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
        assert head_dim % 2 == 0, "Dimension must be divisible by 2"
        # Build the theta parameter
        # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
        # Shape: (Head_Dim / 2)
        theta_numerator = torch.arange(0, head_dim, 2).float()
        # Shape: (Head_Dim / 2)
        theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
        # Construct the positions (the "m" parameter)
        # Shape: (Seq_Len)
        m = torch.arange(seq_len, device=device)
        # Multiply each theta by each position using the outer product.
        # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        freqs = torch.outer(m, theta).float()
        # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
        # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
        freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_complex

    def apply_rotary_embeddings(self, x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
        # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
        # Two consecutive values will become a single complex number
        # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
        # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
        freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
        # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
        # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
        # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
        x_rotated = x_complex * freqs_complex
        # Convert the complex number back to the real number
        # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
        x_out = torch.view_as_real(x_rotated)
        # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
        x_out = x_out.reshape(*x.shape)
        return x_out.type_as(x).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        freqs_complex = self.freqs_complex[:seq_len]
        return self.apply_rotary_embeddings(x, freqs_complex, x.device)

class LSTMAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.head_dim = args.head_dim

        self.w_q = nn.Linear(args.dim, args.head_dim, bias=args.bias)
        self.w_k = nn.Linear(args.dim, args.head_dim, bias=args.bias)
        self.w_v = nn.Linear(args.dim, args.head_dim, bias=args.bias)

        self.dropout = nn.Dropout(args.dropout_p)

        self.out = nn.Linear(args.head_dim, args.dim)

        self.pos_embed = RoPE(args)

        self.register_buffer(
                'causal_mask',
                torch.triu(
                    torch.ones([args.max_seq_len, args.max_seq_len],
                    dtype=torch.bool),
                    diagonal=1
                ).view(1, 1, args.max_seq_len, args.max_seq_len)
            )

    def forward(self, x):
        bs, seq_len, _ = x.shape

        Q, K, V = self.w_q(x), self.w_k(x), self.w_v(x)

        # Reshape resulting tensors
        # (batch_size, seq_len, n_heads * head_dim) -> (batch_size, seq_len, n_heads, head_dim)
        Q = Q.view(bs, seq_len, 1, self.head_dim)
        # (batch_size, seq_len, n_kv_heads * head_dim) -> (batch_size, seq_len, n_kv_heads, head_dim)
        K = K.view(bs, seq_len, 1, self.head_dim)
        V = V.view(bs, seq_len, 1, self.head_dim)

        Q, K = self.pos_embed(Q), self.pos_embed(K)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(2, 3)) / self.head_dim**0.5
        scores.masked_fill_(self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        scores = scores.softmax(dim=-1)
        scores = self.dropout(scores)

        output = torch.matmul(scores, V)
        output = output.transpose(1, 2).reshape(bs, seq_len, -1)
        output = self.out(output)

        return output

class RMSNorm(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.eps = args.norm_eps
        self.g_params = nn.Parameter(torch.ones(args.dim))

    def forward(self, x: torch.Tensor):
        rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return rms * self.g_params

class FeedForward(nn.Module):
    '''FeedForward layer with SwiGLU activation.'''
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.ffn_hidden_dim, bias=False)
        self.w2 = nn.Linear(args.ffn_hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.ffn_hidden_dim, bias=False)

        self.dropout = nn.Dropout(args.dropout_p)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.norm1 = RMSNorm(args)
        self.norm2 = RMSNorm(args)

        self.attention = LSTMAttention(args)

        self.ffn = FeedForward(args)

    def forward(self, x: torch.Tensor):
        h = x + self.attention(self.norm1(x))
        out = h + self.ffn(self.norm2(x))
        return out.mean(1)