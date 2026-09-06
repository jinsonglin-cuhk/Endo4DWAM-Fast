"""Optional real-head CPU regression; run in an environment with DA3 installed."""
import importlib.util
import unittest
import torch

from endo4dwam.models.wan22.helpers.da3_head import build_da3_head
from endo4dwam.models.wan22.helpers.edge_head import build_edge_head


EDGE_WEIGHTS = '/mnt/data2/ljs/Endo4DWAM/EdGE-review/weights/model.safetensors'
EDGE_SOURCE = '/mnt/data2/ljs/Endo4DWAM/EdGE-review'


@unittest.skipUnless(importlib.util.find_spec('depth_anything_3') and importlib.util.find_spec('addict'), 'DA3 optional dependencies unavailable')
class DA3HeadTests(unittest.TestCase):
    def test_pretrained_depth_and_signed_motion(self):
        torch.set_num_threads(2)
        for channels in (None, 2):
            head, _ = build_da3_head(output_dim=channels, device='cpu', dtype=torch.float32, freeze=True)
            self.assertTrue(all(not p.requires_grad for p in head.parameters()))
            if channels:
                self.assertEqual(head.activation, 'linear')
                self.assertEqual(head.out_dim, 3)
                with torch.no_grad():
                    head.scratch.output_conv2[-1].weight.zero_()
                    head.scratch.output_conv2[-1].bias.copy_(torch.tensor([-1., 2., 0.]))
            feats = [[torch.randn(1, 1, 4, 1024, requires_grad=True)] for _ in range(4)]
            output = head(feats, H=64, W=64, patch_start_idx=0)['depth']
            if channels:
                self.assertEqual(tuple(output.shape), (1,1,64,64,2))
                torch.testing.assert_close(output[...,0], torch.full((1,1,64,64), -1.))
                torch.testing.assert_close(output[...,1], torch.full((1,1,64,64), 2.))
            else:
                self.assertEqual(tuple(output.shape), (1,1,64,64))
                output.mean().backward()
                self.assertTrue(any(f[0].grad is not None and f[0].grad.abs().sum() > 0 for f in feats))

    def test_edge_depth_and_signed_motion_readouts(self):
        torch.set_num_threads(2)
        for channels in (None, 2):
            head, _ = build_edge_head(
                EDGE_WEIGHTS, source_path=EDGE_SOURCE, output_dim=channels,
                device='cpu', dtype=torch.float32,
            )
            self.assertEqual(len(head.adapters), 4)
            self.assertEqual(
                head.head.__class__.__module__,
                'edge.models.components.heads.dpt_head',
            )
            probe = torch.randn(2, 1024)
            adapted = head.adapters[0](probe)
            torch.testing.assert_close(adapted[:, :1024], probe)
            torch.testing.assert_close(adapted[:, 1024:], probe)
            feats = [[torch.randn(1, 1, 4, 1024, requires_grad=True)] for _ in range(4)]
            output = head(feats, H=64, W=64, patch_start_idx=0)['depth']
            expected = (1, 1, 64, 64, 2) if channels else (1, 1, 64, 64)
            self.assertEqual(tuple(output.shape), expected)
            output.mean().backward()
            self.assertTrue(all(level[0].grad is not None for level in feats))


if __name__ == '__main__':
    unittest.main()
