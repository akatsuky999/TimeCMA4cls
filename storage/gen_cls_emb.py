"""
Generate text embeddings for time series classification data.

This module generates GPT-2 embeddings for classification datasets,
following the TimeCMA embedding generation approach with:
- Template-based prompt design (similar to TimeCMA's forecasting prompts)
- Natural language descriptions for semantic richness

Key difference from TimeCMA:
- Classification datasets lack explicit timestamps, so we use position-based
  time information: "From step 1 to step N"
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np


PROMPT_TEMPLATE_STATS = (
    "Channel {channel} time series with {seq_len} steps: "
    "mean {mean}, std {std}, min {min}, max {max}, median {median}, "
    "skewness {skewness}, kurtosis {kurtosis}, "
    "trend {trend}, volatility {volatility}, "
    "first-order diff mean {diff_mean}, first-order diff std {diff_std}."
)

PROMPT_TEMPLATE_VALUES = (
    "From step 1 to step {seq_len}, the channel {channel} values were {values}. "
    "The total trend was {trend}."
)


class GenClassificationEmb(nn.Module):
    """
    Generate text embeddings for time series classification data.
    
    This class follows TimeCMA's design philosophy:
    - Template-based prompt generation with position information
    - Natural language descriptions suitable for GPT-2 encoding
    
    Args:
        use_statistics: bool
            - True: PROMPT_TEMPLATE_STATS
            - False: PROMPT_TEMPLATE_VALUES
    """
    
    def __init__(
        self,
        dataset_name='PEMS-SF',
        model_name="gpt2",
        device='cuda:0',
        seq_len=96,
        d_model=768,
        layer=12,
        use_statistics=True,
        max_tokens=896,
    ):
        super(GenClassificationEmb, self).__init__()
        self.dataset_name = dataset_name
        self.device = device
        self.seq_len = seq_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        self.use_statistics = use_statistics
        self.max_tokens = max_tokens
        
        self._sample_ratio = None
        self._calibrated = False
        
        # Load GPT-2 tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)
        
        # Freeze GPT-2 parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
    
    def _compute_statistics(self, values):
        values_np = values.cpu().numpy() if torch.is_tensor(values) else values
  
        non_zero_mask = values_np != 0
        if np.sum(non_zero_mask) > 0:
            values_np = values_np[non_zero_mask]
        
        n = len(values_np)
        
        mean = float(np.mean(values_np))
        std = float(np.std(values_np)) if n > 1 else 0.0
        min_val = float(np.min(values_np))
        max_val = float(np.max(values_np))
        median = float(np.median(values_np))

        if std > 1e-8 and n > 2:
            skewness = float(np.mean(((values_np - mean) / std) ** 3))
        else:
            skewness = 0.0

        if std > 1e-8 and n > 3:
            kurtosis = float(np.mean(((values_np - mean) / std) ** 4) - 3)
        else:
            kurtosis = 0.0
        if n > 1:
            trend = float(np.sum(np.diff(values_np)))
        else:
            trend = 0.0

        volatility = float(std / abs(mean)) if abs(mean) > 1e-8 else 0.0

        if n > 1:
            diff = np.diff(values_np)
            diff_mean = float(np.mean(diff))
            diff_std = float(np.std(diff))
        else:
            diff_mean = 0.0
            diff_std = 0.0
        
        return {
            'seq_len': n,
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'median': median,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'trend': trend,
            'volatility': volatility,
            'diff_mean': diff_mean,
            'diff_std': diff_std,
        }
    
    def _format_stat(self, value):
        if abs(value) >= 100:
            return f"{value:.1f}"
        elif abs(value) >= 1:
            return f"{value:.2f}"
        else:
            return f"{value:.4f}"
    
    def _prepare_prompt_statistics(self, in_data, i, j):
        values = in_data[i, :, j].flatten()
        stats = self._compute_statistics(values)
        
        prompt = PROMPT_TEMPLATE_STATS.format(
            channel=j + 1,
            seq_len=stats['seq_len'],
            mean=self._format_stat(stats['mean']),
            std=self._format_stat(stats['std']),
            min=self._format_stat(stats['min']),
            max=self._format_stat(stats['max']),
            median=self._format_stat(stats['median']),
            skewness=self._format_stat(stats['skewness']),
            kurtosis=self._format_stat(stats['kurtosis']),
            trend=self._format_stat(stats['trend']),
            volatility=self._format_stat(stats['volatility']),
            diff_mean=self._format_stat(stats['diff_mean']),
            diff_std=self._format_stat(stats['diff_std']),
        )
        
        tokenized_prompt = self.tokenizer.encode(
            prompt, 
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(self.device)
        
        return tokenized_prompt
    
    def _compute_trend(self, values):
        """Compute trend value (sum of differences), following TimeCMA."""
        values_np = values.cpu().numpy() if torch.is_tensor(values) else values
        if len(values_np) < 2:
            return 0.0
        return float(np.sum(np.diff(values_np)))
    
    def _format_values_string(self, values, sample_size=None):
        
        values_np = values.cpu().numpy() if torch.is_tensor(values) else values
        
        # Filter out zero-padding
        non_zero_mask = values_np != 0
        if np.sum(non_zero_mask) > 0:
            values_np = values_np[non_zero_mask]
        
        # 等间距采样（如果需要）
        if sample_size is not None and len(values_np) > sample_size:
            indices = np.linspace(0, len(values_np) - 1, sample_size, dtype=int)
            values_np = values_np[indices]
        
        # Format as integers (following TimeCMA style) or with 2 decimals for small values
        if np.max(np.abs(values_np)) > 10:
            values_str = ", ".join([str(int(v)) for v in values_np])
        else:
            values_str = ", ".join([f"{v:.2f}" for v in values_np])
        
        return values_str, len(values_np)
    
    def _calibrate_sampling(self, in_data):
        if self._calibrated or self.use_statistics:
            return

        values = in_data[0, :, 0].flatten()
        values_np = values.cpu().numpy() if torch.is_tensor(values) else values

        non_zero_mask = values_np != 0
        if np.sum(non_zero_mask) > 0:
            values_np = values_np[non_zero_mask]
        
        original_len = len(values_np)

        values_str, _ = self._format_values_string(values, sample_size=None)
        trend = self._compute_trend(values)
        trend_str = f"{trend:.0f}" if abs(trend) > 1 else f"{trend:.2f}"
        
        test_prompt = PROMPT_TEMPLATE_VALUES.format(
            seq_len=original_len,
            channel=1,
            values=values_str,
            trend=trend_str,
        )
        
        token_count = len(self.tokenizer.encode(test_prompt))
        
        if token_count <= self.max_tokens:
            self._sample_ratio = None
            print(f"[Calibration] Token count: {token_count} <= {self.max_tokens}, using full sequence")
        else:
            template_overhead = 40
            tokens_per_value = (token_count - template_overhead) / original_len
            target_values = int((self.max_tokens - template_overhead) / tokens_per_value)
            target_values = max(10, min(target_values, original_len))
            low, high = 10, original_len
            best_sample_size = target_values
            
            while low <= high:
                mid = (low + high) // 2
                test_str, _ = self._format_values_string(values, sample_size=mid)
                test_prompt = PROMPT_TEMPLATE_VALUES.format(
                    seq_len=mid,
                    channel=1,
                    values=test_str,
                    trend=trend_str,
                )
                test_tokens = len(self.tokenizer.encode(test_prompt))
                
                if test_tokens <= self.max_tokens:
                    best_sample_size = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            self._sample_ratio = best_sample_size

            final_str, _ = self._format_values_string(values, sample_size=best_sample_size)
            final_prompt = PROMPT_TEMPLATE_VALUES.format(
                seq_len=best_sample_size,
                channel=1,
                values=final_str,
                trend=trend_str,
            )
            final_tokens = len(self.tokenizer.encode(final_prompt))
            
            print(f"[Calibration] Original: {original_len} points, {token_count} tokens")
            print(f"[Calibration] Sampling to {best_sample_size} points, {final_tokens} tokens (<= {self.max_tokens})")
        
        self._calibrated = True
    
    def _prepare_prompt_values(self, in_data, i, j):
        """Prepare prompt with dynamic sampling to ensure tokens <= 1024 (GPT-2 max)."""
        values = in_data[i, :, j].flatten()
        trend = self._compute_trend(values)
        trend_str = f"{trend:.0f}" if abs(trend) > 1 else f"{trend:.2f}"
        
        # Start with calibrated sample size or full sequence
        current_sample_size = self._sample_ratio
        max_model_tokens = 1024  # GPT-2 hard limit
        
        # Generate prompt and check token count
        values_str, valid_len = self._format_values_string(values, sample_size=current_sample_size)
        prompt = PROMPT_TEMPLATE_VALUES.format(
            seq_len=valid_len,
            channel=j + 1,
            values=values_str,
            trend=trend_str,
        )
        
        token_count = len(self.tokenizer.encode(prompt))
        
        # If still exceeds GPT-2 max, progressively reduce sample size
        if token_count > max_model_tokens:
            # Binary search for optimal sample size for this specific sample
            values_np = values.cpu().numpy() if torch.is_tensor(values) else values
            non_zero_mask = values_np != 0
            if np.sum(non_zero_mask) > 0:
                original_len = int(np.sum(non_zero_mask))
            else:
                original_len = len(values_np)
            
            low, high = 10, current_sample_size if current_sample_size else original_len
            best_sample_size = low
            
            while low <= high:
                mid = (low + high) // 2
                test_str, test_len = self._format_values_string(values, sample_size=mid)
                test_prompt = PROMPT_TEMPLATE_VALUES.format(
                    seq_len=test_len,
                    channel=j + 1,
                    values=test_str,
                    trend=trend_str,
                )
                test_tokens = len(self.tokenizer.encode(test_prompt))
                
                if test_tokens <= max_model_tokens:
                    best_sample_size = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            # Use the best sample size found
            values_str, valid_len = self._format_values_string(values, sample_size=best_sample_size)
            prompt = PROMPT_TEMPLATE_VALUES.format(
            seq_len=valid_len,
            channel=j + 1,
            values=values_str,
            trend=trend_str,
        )
        
        # Final tokenization with truncation as safety net (should not trigger)
        tokenized_prompt = self.tokenizer.encode(
            prompt, 
            return_tensors="pt",
            max_length=max_model_tokens,
            truncation=True,
        ).to(self.device)
        
        return tokenized_prompt
    
    def _prepare_prompt(self, in_data, i, j):
        if self.use_statistics:
            return self._prepare_prompt_statistics(in_data, i, j)
        else:
            return self._prepare_prompt_values(in_data, i, j)
    
    def forward(self, tokenized_prompt):
        """Get embeddings from tokenized prompt."""
        with torch.no_grad():
            outputs = self.model(tokenized_prompt)
            prompt_embeddings = outputs.last_hidden_state
        return prompt_embeddings
    
    def generate_embeddings(self, in_data):
        """
        Generate embeddings for a batch of time series samples.
        
        Args:
            in_data: Input data tensor [B, L, N] (batch, seq_len, features)
            
        Returns:
            embeddings: Tensor of embeddings [B, E, N] where E=d_model (768)
        """
        B, L, N = in_data.shape
        
        if not self.use_statistics and not self._calibrated:
            self._calibrate_sampling(in_data)
        
        tokenized_prompts = []
        max_token_count = 0
        
        for i in range(B):
            for j in range(N):
                tokenized_prompt = self._prepare_prompt(in_data, i, j)
                max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                tokenized_prompts.append((i, tokenized_prompt, j))
        
        in_prompt_emb = torch.zeros(
            (B, max_token_count, self.d_model, N), 
            dtype=torch.float32, 
            device=self.device
        )
        
        for i, tokenized_prompt, j in tokenized_prompts:
            prompt_emb = self.forward(tokenized_prompt)
            
            padding_length = max_token_count - tokenized_prompt.shape[1]
            if padding_length > 0:
                last_token_emb = prompt_emb[:, -1, :].unsqueeze(1)
                padding = last_token_emb.repeat(1, padding_length, 1)
                prompt_emb_padded = torch.cat([prompt_emb, padding], dim=1)
            else:
                prompt_emb_padded = prompt_emb
            
            in_prompt_emb[i, :, :, j] = prompt_emb_padded.squeeze(0)
        
        last_token_emb = in_prompt_emb[:, max_token_count-1, :, :]
        
        return last_token_emb
    
    def generate_single_embedding(self, sample_data):
        """
        Generate embedding for a single sample.
        
        Args:
            sample_data: Single sample tensor [L, N] (seq_len, features)
            
        Returns:
            embedding: Tensor [E, N] where E=d_model (768)
        """
        sample_data = sample_data.unsqueeze(0)
        embeddings = self.generate_embeddings(sample_data)
        return embeddings.squeeze(0)


class GenClassificationEmbSimple(nn.Module):
    """
    Simplified embedding generator using statistical features.
    
    This is faster than full prompt encoding and still provides semantic value.
    """
    
    def __init__(
        self,
        dataset_name='PEMS-SF',
        model_name="gpt2",
        device='cuda:0',
        d_model=768
    ):
        super(GenClassificationEmbSimple, self).__init__()
        self.device = device
        self.d_model = d_model
        self.dataset_name = dataset_name
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self._init_template_embeddings()
    
    def _init_template_embeddings(self):
        """Pre-compute template embeddings for statistical descriptions."""
        templates = [
            "mean value",
            "standard deviation",
            "minimum value",
            "maximum value",
            "trend direction",
            "value range"
        ]
        
        self.stat_embeddings = []
        for template in templates:
            tokens = self.tokenizer.encode(template, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model(tokens).last_hidden_state[0, -1, :]
            self.stat_embeddings.append(emb)
        
        self.stat_embeddings = torch.stack(self.stat_embeddings, dim=0)
    
    def generate_embeddings(self, in_data):
        """
        Generate embeddings based on statistical features.
        
        Args:
            in_data: Input data tensor [B, L, N]
            
        Returns:
            embeddings: Tensor [B, E, N]
        """
        B, L, N = in_data.shape
        
        mean_vals = in_data.mean(dim=1, keepdim=True)
        std_vals = in_data.std(dim=1, keepdim=True)
        min_vals = in_data.min(dim=1, keepdim=True)[0]
        max_vals = in_data.max(dim=1, keepdim=True)[0]
        
        diff = in_data[:, 1:, :] - in_data[:, :-1, :]
        trend_vals = diff.sum(dim=1, keepdim=True)
        
        range_vals = max_vals - min_vals
        
        stats = torch.cat([mean_vals, std_vals, min_vals, max_vals, trend_vals, range_vals], dim=1)
        stats = (stats - stats.mean(dim=1, keepdim=True)) / (stats.std(dim=1, keepdim=True) + 1e-8)
        
        embeddings = torch.einsum('bsn,se->ben', stats, self.stat_embeddings)
        embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        
        return embeddings
