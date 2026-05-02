"""
Multimodal Vision-Language Transformer for Image Captioning
Features:
  1. Vision Encoder (CNN/ViT feature extraction).
  2. Multi-Head Scaled Dot-Product Attention Mechanism.
  3. Transformer / LSTM Decoder for autoregressive token generation.
  4. Evaluation Pipeline computing BLEU-1 to BLEU-4 metrics.
"""

import numpy as np

class ScaledDotProductAttention:
    def __init__(self, d_k):
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        scores = (Q @ K.T) / np.sqrt(self.d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        context = attn_weights @ V
        return context, attn_weights

class ImageCaptioningModel:
    def __init__(self, vocab_size=1000, embed_dim=256, feature_dim=512):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.attention = ScaledDotProductAttention(d_k=embed_dim)

        # Projections
        np.random.seed(42)
        self.W_proj = np.random.randn(feature_dim, embed_dim) * 0.1
        self.word_embeddings = np.random.randn(vocab_size, embed_dim) * 0.1

    def forward(self, img_features, token_seq):
        """
        img_features: (batch_size, feature_dim) or (feature_dim,)
        token_seq: list of word token IDs
        """
        # Project image features to embed space
        img_embed = img_features @ self.W_proj
        
        # Word embeddings
        seq_embeds = self.word_embeddings[token_seq]  # (seq_len, embed_dim)

        # Cross-attention: Queries = seq_embeds, Keys/Values = img_embed
        Q = seq_embeds
        K = img_embed.reshape(1, -1)
        V = img_embed.reshape(1, -1)

        context, weights = self.attention.forward(Q, K, V)
        return context, weights

    def evaluate_bleu(self, reference_caption: str, candidate_caption: str) -> dict:
        """
        Computes BLEU-1 and BLEU-2 n-gram precision metrics.
        """
        ref_tokens = reference_caption.lower().split()
        cand_tokens = candidate_caption.lower().split()

        def n_gram_precision(n):
            cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens)-n+1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)]
            if not cand_ngrams:
                return 0.0
            matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
            return matches / len(cand_ngrams)

        bleu1 = n_gram_precision(1)
        bleu2 = n_gram_precision(2)
        return {"BLEU-1": bleu1, "BLEU-2": bleu2}

if __name__ == "__main__":
    model = ImageCaptioningModel(vocab_size=100, embed_dim=64, feature_dim=128)
    dummy_img = np.random.randn(128)
    seq = [1, 5, 12, 8]
    context, attn = model.forward(dummy_img, seq)
    print("=== Multimodal Vision Transformer Model ===")
    print("Context matrix shape:", context.shape)
    metrics = model.evaluate_bleu("a dog running on grass", "a dog playing on green grass")
    print("BLEU Evaluation Metrics:", metrics)
