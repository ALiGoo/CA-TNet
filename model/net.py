import torch
import torch.nn as nn
from torch import Tensor
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:,:x.size(1)]
        return x


class PitchEmbeddingWithWord(torch.nn.Module):

    def __init__(self, d_model, dropout_rate):
        super(PitchEmbeddingWithWord, self).__init__()
        self.pitch_embeddings = nn.Linear(1, d_model)
        self.syllable_token_embeddings = nn.Embedding(5, d_model, padding_idx=0)
        self.syllable_segment_embeddings = nn.Embedding(2, d_model, padding_idx=0)
        self.word_token_embeddings = nn.Embedding(6, d_model, padding_idx=0)
        self.word_segment_embeddings = nn.Embedding(2, d_model, padding_idx=0)
        self.position_embeddings = PositionalEncoding(d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, f0, syllable_token, syllable_boundary, word_token, word_boundary):
        x = self.pitch_embeddings(f0)
        x = x + self.syllable_token_embeddings(syllable_token)
        x = x + self.syllable_segment_embeddings(syllable_boundary)
        x = x + self.word_token_embeddings(word_token)
        x = x + self.word_segment_embeddings(word_boundary)
        x = self.position_embeddings(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x


class RhythmEmbedding(torch.nn.Module):

    def __init__(self, d_in, d_model, dropout_rate):
        super(RhythmEmbedding, self).__init__()
        self.rhythm_embeddings = nn.Linear(d_in, d_model)
        self.token_embeddings = nn.Embedding(2, d_model, padding_idx=0)
        self.position_embeddings = PositionalEncoding(d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, syllable_duration_feature, syllable_duration_token):
        x = self.rhythm_embeddings(syllable_duration_feature)
        x = x + self.token_embeddings(syllable_duration_token)
        x = self.position_embeddings(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x


class SyllableEmbedding(torch.nn.Module):

    def __init__(self, d_model, dropout_rate):
        super(SyllableEmbedding, self).__init__()
        self.position_embeddings = PositionalEncoding(d_model)
        self.layernorm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.position_embeddings(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x


class ToneNetWithWordEmbeddingWithRhythm(torch.nn.Module):

    def __init__(self, d_rhythm_in, d_model, nhead=4, dropout_rate=0.1):
        super(ToneNetWithWordEmbeddingWithRhythm, self).__init__()
        self.pitch_embedding = PitchEmbeddingWithWord(d_model, dropout_rate)
        self.rhythm_embedding = RhythmEmbedding(d_rhythm_in, d_model, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout_rate,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True
        )
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout_rate,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True
        )
        self.pitch_encoder = nn.TransformerEncoder(encoder_layers, 4)
        self.rhythm_encoder = nn.TransformerEncoder(encoder_layers, 2)
        self.syllable_embedding = SyllableEmbedding(d_model, dropout_rate)
        self.syllable_encoder = nn.TransformerDecoder(decoder_layers, 2)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(d_model, 5),
        )

        self.apply(self._init_weights)

    def forward(self, f0, syllable_token, syllable_boundary, word_token, word_boundary, syllable_idx, syllable_duration_feature, syllable_duration_token):
        x = self.pitch_embedding(f0, syllable_token, syllable_boundary, word_token, word_boundary)
        x = self.pitch_encoder(x, src_key_padding_mask=syllable_token==0)
        
        x_rhythm = self.rhythm_embedding(syllable_duration_feature, syllable_duration_token)
        x_rhythm = self.rhythm_encoder(x_rhythm, src_key_padding_mask=syllable_duration_token==0)

        x = x[torch.arange(x.size(0)).unsqueeze(1), syllable_idx] * (syllable_idx!=0)[:,:, None].float()
        x = self.syllable_embedding(x)
        x = self.syllable_encoder(
            tgt=x,
            memory=x_rhythm,
            tgt_key_padding_mask=syllable_idx==-1,
            memory_key_padding_mask=syllable_duration_token==0,
        )

        logits = self.classifier(x)

        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
            if module.padding_idx is not None:
                nn.init.constant_(module.weight[module.padding_idx], 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
