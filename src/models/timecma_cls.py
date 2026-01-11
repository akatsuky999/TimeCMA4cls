"""
TimeCMA for Time Series Classification.
Adapted from the original TimeCMA prediction model to support classification tasks.

Key modifications:
1. Changed output layer from prediction to classification
2. Added classification head with multi-head pooling
3. Adapted cross-modal alignment for classification features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .layers.StandardNorm import Normalize
from .layers.Cross_Modal_Align import CrossModal


class TimeCMAClassifier(nn.Module):
    """
    TimeCMA model adapted for time series classification.
    
    This implementation follows the EXACT same architecture as original TimeCMA,
    only replacing the prediction decoder with a classification head.
    
    Original TimeCMA structure:
    1. RevIN normalization
    2. Linear projection (seq_len -> channel)
    3. Time Series Encoder (TransformerEncoder)
    4. Prompt Encoder (TransformerEncoder)  
    5. Cross-Modal Alignment
    6. [Original: Decoder + Projection] -> [Classification: Flatten + Classifier]
    """
    
    def __init__(self, config, data):
        super().__init__()
        
        # Basic configuration (same as original TimeCMA)
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.num_nodes = data.feature_df.shape[1]  # 原始TimeCMA叫num_nodes
        self.num_classes = len(data.class_names)
        
        # Model hyperparameters (matching original TimeCMA defaults)
        self.channel = config.get('channel', 32)      # 原始默认32
        self.d_llm = config.get('d_llm', 768)
        self.dropout = config.get('dropout', 0.1)
        self.head = config.get('n_heads', 8)          # 原始叫head
        self.e_layer = config.get('e_layers', 1)      # 原始默认1层
        self.d_layer = config.get('d_layers', 1)      # 原始默认1层 (decoder层数)
        self.d_ff = config.get('d_ff', 32)            # 原始默认32
        
        # ============ 以下完全复制原始 TimeCMA 结构 ============
        
        # RevIN normalization (SAME AS ORIGINAL)
        self.normalize_layers = Normalize(self.num_nodes, affine=False)
        
        # Input projection: seq_len -> channel (SAME AS ORIGINAL)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel)
        
        # Time Series Encoder (SAME AS ORIGINAL - 不指定 dim_feedforward，使用默认2048)
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, 
            nhead=self.head, 
            batch_first=True,
            norm_first=True, 
            dropout=self.dropout
        )
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer, 
            num_layers=self.e_layer
        )
        
        # Prompt Encoder (SAME AS ORIGINAL)
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, 
            nhead=self.head, 
            batch_first=True,
            norm_first=True, 
            dropout=self.dropout
        )
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer, 
            num_layers=self.e_layer
        )
        
        # Cross-Modal Alignment (SAME AS ORIGINAL)
        self.cross = CrossModal(
            d_model=self.num_nodes,  # 原始用 num_nodes
            n_heads=1,
            d_ff=self.d_ff,
            norm='LayerNorm',
            attn_dropout=self.dropout,
            dropout=self.dropout,
            pre_norm=True,
            activation="gelu",
            res_attention=True,
            n_layers=1,
            store_attn=False
        )
        
        # Transformer Decoder (SAME AS ORIGINAL - 特征增强模块)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel,
            nhead=self.head,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout
        )
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=self.d_layer
        )
        
        # ============ 分类任务：替换 c_to_length 为 Classifier ============
        # 原始 TimeCMA: c_to_length (Linear: channel -> pred_len) 用于预测
        # 分类任务: flatten + classifier
        
        self.flatten_dim = self.channel * self.num_nodes
        self.classifier = nn.Linear(self.flatten_dim, self.num_classes)
        
    def forward(self, x_enc, x_mark_enc=None, embeddings=None):
        """
        Forward pass - follows EXACT same flow as original TimeCMA.
        
        Original TimeCMA forward flow:
        1. RevIN norm
        2. permute [B,L,N] -> [B,N,L]
        3. length_to_feature [B,N,L] -> [B,N,C]
        4. ts_encoder [B,N,C]
        5. permute [B,N,C] -> [B,C,N]
        6. embeddings: squeeze, permute [B,E,N] -> [B,N,E]
        7. prompt_encoder [B,N,E]
        8. permute [B,N,E] -> [B,E,N]
        9. cross [B,C,N] x [B,E,N] -> [B,C,N]
        10. permute [B,C,N] -> [B,N,C]
        11. decoder(cross_out, cross_out) -> [B,N,C]  (SAME AS ORIGINAL)
        12. [Original: c_to_length] -> [Classification: flatten + classify]
        """
        B = x_enc.shape[0]
        input_data = x_enc.float()
        
        # TimeCMA requires embeddings
        if embeddings is None:
            raise ValueError("TimeCMA requires embeddings for cross-modal alignment.")
        
        # ===== 完全复制原始 TimeCMA forward 流程 =====
        
        # RevIN
        input_data = self.normalize_layers(input_data, 'norm')
        
        input_data = input_data.permute(0, 2, 1)  # [B, N, L]
        input_data = self.length_to_feature(input_data)  # [B, N, C]
        
        emb = embeddings.float()
        if emb.dim() == 4:
            emb = emb.squeeze(-1)  # [B, E, N]
        emb = emb.permute(0, 2, 1)  # [B, N, E]
        
        # Encoder
        enc_out = self.ts_encoder(input_data)  # [B, N, C]
        enc_out = enc_out.permute(0, 2, 1)  # [B, C, N]
        emb = self.prompt_encoder(emb)  # [B, N, E]
        emb = emb.permute(0, 2, 1)  # [B, E, N]
        
        # Cross-modal alignment
        cross_out = self.cross(enc_out, emb, emb)  # [B, C, N]
        cross_out = cross_out.permute(0, 2, 1)  # [B, N, C]
        
        # Decoder (SAME AS ORIGINAL - 特征增强)
        dec_out = self.decoder(cross_out, cross_out)  # [B, N, C]
        
        # ===== 分类任务：替换 c_to_length 为 classifier =====
        # 原始: dec_out = self.c_to_length(dec_out) -> [B, N, pred_len]
        # 分类: flatten + classify
        
        features = dec_out.reshape(B, -1)  # [B, N*C]
        logits = self.classifier(features)
        
        return logits
    
    def count_parameters(self, trainable=True):
        """Count model parameters."""
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class TimeCMAClassifierWithPatching(nn.Module):
    """
    TimeCMA classifier with patch-based input processing.
    
    This version processes the time series using patches (similar to GPT4TS),
    then applies cross-modal alignment with text embeddings.
    """
    
    def __init__(self, config, data):
        super().__init__()
        
        # Basic configuration
        self.seq_len = data.max_seq_len
        self.max_len = data.max_seq_len
        self.feat_dim = data.feature_df.shape[1]
        self.num_classes = len(data.class_names)
        
        # Model hyperparameters
        self.d_model = config.get('d_model', 768)
        self.patch_size = config.get('patch_size', 8)
        self.stride = config.get('stride', 8)
        self.d_llm = config.get('d_llm', 768)
        self.dropout = config.get('dropout', 0.1)
        self.n_heads = config.get('n_heads', 8)
        self.e_layers = config.get('e_layers', 2)
        self.d_ff = config.get('d_ff', 256)
        
        # Calculate number of patches
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        
        # Normalization
        self.normalize_layers = Normalize(self.feat_dim, affine=False)
        
        # Patch embedding: embed each patch
        patch_dim = self.feat_dim * self.patch_size
        self.patch_embedding = nn.Linear(patch_dim, self.d_model)
        
        # Positional encoding for patches
        self.pos_encoding = nn.Parameter(torch.randn(1, self.patch_num, self.d_model) * 0.02)
        
        # Time Series Encoder (same as original TimeCMA)
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout,
            dim_feedforward=self.d_ff * 4
        )
        self.ts_encoder = nn.TransformerEncoder(
            self.ts_encoder_layer,
            num_layers=self.e_layers
        )
        
        # Prompt/Embedding Encoder (SAME AS ORIGINAL TimeCMA!)
        # This is critical - original TimeCMA uses a full TransformerEncoder, not just projection
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm,
            nhead=self.n_heads,
            batch_first=True,
            norm_first=True,
            dropout=self.dropout,
            dim_feedforward=self.d_ff * 4
        )
        self.prompt_encoder = nn.TransformerEncoder(
            self.prompt_encoder_layer,
            num_layers=self.e_layers
        )
        
        # Project encoded embeddings to model dimension for cross-modal alignment
        self.emb_projection = nn.Linear(self.d_llm, self.d_model)
        
        # Cross-modal alignment - THE CORE OF TimeCMA
        self.cross_modal = CrossModal(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            norm='LayerNorm',
            attn_dropout=self.dropout,
            dropout=self.dropout,
            pre_norm=True,
            activation="gelu",
            res_attention=True,
            n_layers=1
        )
        
        # Classification head
        self.act = nn.GELU()
        self.dropout_layer = nn.Dropout(self.dropout)
        self.ln_proj = nn.LayerNorm(self.d_model * self.patch_num)
        self.classifier = nn.Linear(self.d_model * self.patch_num, self.num_classes)
        
    def forward(self, x_enc, x_mark_enc=None, embeddings=None):
        """
        Forward pass with patching.
        
        Args:
            x_enc: Input time series [B, L, N]
            x_mark_enc: Time marks (optional)
            embeddings: Text embeddings [B, E, N]
            
        Returns:
            logits: Classification logits [B, num_classes]
        """
        B, L, N = x_enc.shape
        x = x_enc.float()
        
        # RevIN normalization
        x = self.normalize_layers(x, 'norm')
        
        # Create patches: [B, L, N] -> [B, N, L] -> pad -> unfold -> [B, N, num_patches, patch_size]
        x = rearrange(x, 'b l n -> b n l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        
        # Flatten patches: [B, N, P, S] -> [B, P, N*S]
        x = rearrange(x, 'b n p s -> b p (s n)')
        
        # Patch embedding: [B, P, N*S] -> [B, P, D]
        x = self.patch_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Time series encoding: [B, P, D]
        enc_out = self.ts_encoder(x)
        
        # TimeCMA requires embeddings for cross-modal alignment
        if embeddings is None:
            raise ValueError(
                "TimeCMA model requires embeddings for cross-modal alignment. "
                "Please generate embeddings first using: "
                "python scripts/generate_embeddings.py --dataset YOUR_DATASET --split train/test"
            )
        
        # Process embeddings (SAME AS ORIGINAL TimeCMA)
        emb = embeddings.float()
        if emb.dim() == 4:
            emb = emb.squeeze(-1)
        
        # [B, E, N] -> [B, N, E]
        emb = emb.permute(0, 2, 1)
        
        # Encode embeddings with Transformer (SAME AS ORIGINAL TimeCMA!)
        # This is the key step that original TimeCMA uses
        emb_encoded = self.prompt_encoder(emb)  # [B, N, E]
        
        # Project to model dimension: [B, N, E] -> [B, N, D]
        emb_projected = self.emb_projection(emb_encoded)
        
        # Cross-modal alignment - THE CORE OF TimeCMA
        # Q=enc_out [B, P, D], K/V=emb_projected [B, N, D]
        cross_out = self.cross_modal(enc_out, emb_projected, emb_projected)
        
        # Flatten for classification: [B, P, D] -> [B, P*D]
        features = cross_out.reshape(B, -1)
        
        # Classification
        features = self.ln_proj(features)
        features = self.act(features)
        features = self.dropout_layer(features)
        logits = self.classifier(features)
        
        return logits
    
    def count_parameters(self, trainable=True):
        """Count model parameters."""
        if trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Factory function to create model
def timecma_factory(config, data):
    """
    Factory function to create TimeCMA classifier.
    
    Args:
        config: Configuration dictionary
        data: Data object
        
    Returns:
        TimeCMA classifier model
    """
    use_patching = config.get('use_patching', True)
    
    if use_patching:
        return TimeCMAClassifierWithPatching(config, data)
    else:
        return TimeCMAClassifier(config, data)

