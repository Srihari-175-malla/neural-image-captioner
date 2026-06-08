import unittest
import sys, os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from caption_model import ImageCaptioningModel

class TestImageCaptioning(unittest.TestCase):
    def setUp(self):
        self.model = ImageCaptioningModel(vocab_size=100, embed_dim=32, feature_dim=64)

    def test_forward_pass(self):
        img_feats = np.random.randn(64)
        seq = [2, 10, 15]
        ctx, attn = self.model.forward(img_feats, seq)
        self.assertEqual(ctx.shape, (3, 32))
        self.assertEqual(attn.shape, (3, 1))

    def test_bleu_metric(self):
        ref = "a red car parked on the street"
        cand = "a red car on street"
        res = self.model.evaluate_bleu(ref, cand)
        self.assertGreater(res["BLEU-1"], 0.5)

if __name__ == '__main__':
    unittest.main()
