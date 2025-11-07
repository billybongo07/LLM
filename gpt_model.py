import torch
import torch.nn as nn
import tiktoken 

#to instantiate; model = GPTModel(GPT_CONFIG_124M)
#takes batches of input from tokenizer --> dataloader to batch sequences --> model

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size (number of tokens in tiktoken) without it the token embedding layer (nn.Embedding) and output head (nn.Linear) couldn't map tokens to vectors or predict next tokens.
    "context_length": 1024, # (from above) specifically the model wouldnt know how to create an embedding matrix large enough to handle all possible tokens.
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
# Input: A tensor x (typically of shape (batch_size, seq_len, emb_dim) from first linear in feedforward
# Output: A tensor of the same shape as x, with the GELU activation goes to second linear in feed forward
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )
# input a tensor x; from layer norm 2 in transformer block
#  output a tensor with the same shape as x after passing through two linear layers with gelu in between
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
#input a tensor x from LayerNorm 1 (transformerblock.forward())
#output a tensor of context vectors
#context-aware representations after multi-head self-attention with causal masking and dropout.
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`, 
        # this will result in errors in the mask creation further below. 
        # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs  
        # do not exceed `context_length` before reaching this forward method.

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) #creates token embeddings for each token in vocab_size with output dim = emb_dim
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
#input a tensor x, Comes from text_to_token_ids() or generate_text_simple()
#output a tensor of logits goes into the first transformer block
#logits are a raw unormalized prediction scores for each token position, used for next-token prediction. (produced by final output layer)
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)      #final layer normalization 
        logits = self.out_head(x)   #linear(768 -> 50257)
        return logits


class TransformerBlock( nn.Module ):
    def __init__( self, cfg ):
        super().__init__()
        self.norm1 = LayerNorm( cfg["emb_dim"] )
        self.att = MultiHeadAttention( 
            d_in = cfg["emb_dim"], 
            d_out = cfg["emb_dim"], 
            context_length = cfg["context_length"], 
            dropout = cfg["drop_rate"], 
            num_heads = cfg["n_heads"] 
        )
        self.dropout = nn.Dropout( cfg["drop_rate"] )
        self.norm2 = LayerNorm( cfg["emb_dim"] )
        self.feedforward = FeedForward( cfg )
                #token and positional embed layer --> dropout --> [layer norm1 --> mask MHA --> dropout --> feed back input (shortcut) --> 
                # layer norm2 -->  feed forward --> dropout --> feedback input (shortcut).] --> repeat x12 --> final layer norm and linear output layer.
    #input a tensor x from GPTModel.forward()
    #output: a tensor with shame shape after applying layer norm, MHA, dropout, shortcut, another layer norm, feed forward, dropout, shortcut
    # ie output is new tensor with origional x tensor added on
    def forward( self, x ):
        
        # the attention chunk in this transformer:
        shortcut = x            #save after attention
        x = self.norm1( x )     # layer norm 1
        x = self.att( x )       #MHA.forward
        x = self.dropout( x )   
        x = x + shortcut        #adds on origional input tensor to x
        
        # the feed forward chunk:
        shortcut = x             
        x = self.norm2( x ) 
        x = self.feedforward( x )    #feedforward.forward(x)
        x = self.dropout( x )
        x = x + shortcut 
        
        return x 

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
#input a tensor x
#output: tensor of same shape, normalized (zscores) across the last dimendsion with learnable scale and shift
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

#input: model: instance of GPTModel, idx: tensor (shape: (batch_size, n_tokens), initial token indices), max_new_tokens: int num of new tokens to gen, context_size: int max context length to consider
#output: a tensor of the extended sequence with generated tokens
#feeds back into GPTModel.forward again (autoregressive)
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

#input: token ids: a tensor (shape: (1, num_tokens), token IDs with batch dimension), tokenizer: tiktoken instance
#output: string decoded from the token IDS
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


#Function / Layer,                Input x comes from,                           Output goes to

#GPTModel.forward,                in_idx from tokenizer or generator,           logits (vocab scores)
#Token + Pos Embedding,           in_idx → embedding layers,                    x → first TransformerBlock
#TransformerBlock.forward,        x from previous block (or embedding dropout), x → next block (12×)
#LayerNorm 1,                     x from block input,                           MultiHeadAttention
#MultiHeadAttention.forward,      output of LayerNorm 1,                        residual add → LayerNorm 2
#ayerNorm 2,                      output after attention residual,              FeedForward
#eedForward.forward,              output of LayerNorm 2,                        residual add → next block
#First Linear in FeedForward,     x from LayerNorm 2,                           GELU
#GELU.forward,                    First Linear in FeedForward,                  Second Linear in FeedForward
#Final LayerNorm,                 output of last TransformerBlock,              Final Linear head
#Final Linear (out_head),         output of final LayerNorm,                    logits → softmax → next token