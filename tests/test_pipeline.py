import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from types import MethodType
import unittest

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from endo4dwam.datasets.lerobot.geometry import GeometryLabels
from endo4dwam.datasets.dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop
from endo4dwam.models.wan22.geometry_losses import clip_shared_affine_depth_loss, flow_loss
from endo4dwam.models.wan22.geometry_branch import build_geometry_branch
from endo4dwam.models.wan22.mot import MoT, scale_gradient
from endo4dwam.models.wan22.persistent_memory import PersistentWorldMemory
from endo4dwam.models.wan22.wan_video_dit import WanVideoDiT
from scripts.val_chunk_endowam import infer_action_window, load_training_config

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_declared_dependencies_cover_enabled_training_features(self):
        project = (ROOT / "pyproject.toml").read_text()
        self.assertIn('"addict==', project)
        self.assertIn('"peft==', project)

    def test_all_clean_history_latents_use_zero_timestep(self):
        dit = WanVideoDiT(
            hidden_dim=8,
            in_dim=2,
            ffn_dim=16,
            out_dim=2,
            text_dim=4,
            freq_dim=4,
            eps=1e-6,
            patch_size=(1, 1, 1),
            num_heads=2,
            attn_head_dim=4,
            num_layers=0,
            has_image_input=False,
            seperated_timestep=True,
            fuse_vae_embedding_in_latents=True,
            video_attention_mask_mode="first_frame_causal",
            num_history_latent_frames=2,
        )
        pre = dit.pre_dit(
            x=torch.randn(1, 2, 3, 1, 1),
            timestep=torch.tensor([500.0]),
            context=torch.randn(1, 2, 4),
            context_mask=torch.ones(1, 2, dtype=torch.bool),
            fuse_vae_embedding_in_latents=True,
        )
        torch.testing.assert_close(pre["t"][:, 0], pre["t"][:, 1])
        self.assertFalse(torch.allclose(pre["t"][:, 1], pre["t"][:, 2]))

    def test_action_only_inference_accepts_k_history_attention_alias(self):
        from endo4dwam.models.wan22.endo4dwam import Endo4DWAM

        model = Endo4DWAM.__new__(Endo4DWAM)
        torch.nn.Module.__init__(model)
        model.video_expert = torch.nn.Module()
        model.video_expert.video_attention_mask_mode = "first_k_frames_causal"
        model.video_expert.num_history_latent_frames = 2
        model.vae = SimpleNamespace(temporal_downsample_factor=4)
        with self.assertRaisesRegex(ValueError, "input_image.*5 history frame"):
            model.infer_action(prompt=None, input_image=torch.empty(0), action_horizon=1)

    def test_fastwam_baseline_and_ablation_axes_are_orthogonal(self):
        from endo4dwam.runtime import _validate_pipeline_config

        with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'configs')):
            baseline = compose(
                config_name='train',
                overrides=['task=endowam_fastwam_baseline_1cam_1e-4'],
            )
            k2 = compose(
                config_name='train',
                overrides=['task=endowam_fastwam_baseline_1cam_1e-4', 'history=k2'],
            )
            cached = compose(
                config_name='train',
                overrides=[
                    'task=endowam_fastwam_baseline_1cam_1e-4',
                    'history=k2',
                    'persistent_memory=cached_control',
                ],
            )
            memory = compose(
                config_name='train',
                overrides=[
                    'task=endowam_fastwam_baseline_1cam_1e-4',
                    'history=k2',
                    'persistent_memory=on',
                ],
            )
            geometry = compose(
                config_name='train',
                overrides=[
                    'task=endowam_fastwam_baseline_1cam_1e-4',
                    'history=k2',
                    'auxiliary=geometry',
                    'persistent_memory=cached_control',
                ],
            )

        self.assertFalse(baseline.eval_generate_video)
        self.assertFalse(baseline.model.lora.enable)
        self.assertEqual(baseline.model.training_attention_path, 'mixed')
        self.assertFalse(baseline.model.memory.enable)
        self.assertFalse(baseline.model.geometry.enable)
        self.assertEqual(baseline.data.train.num_frames, 33)
        self.assertEqual(baseline.data.train.num_frames - 1, 32)

        self.assertEqual(k2.model.video_dit_config.num_history_latent_frames, 2)
        self.assertEqual(k2.data.train.num_frames, 41)
        history_actions = 4 * (k2.model.video_dit_config.num_history_latent_frames - 1) \
            * k2.data.train.action_video_freq_ratio
        self.assertEqual(k2.data.train.num_frames - 1 - history_actions, 32)
        self.assertFalse(k2.model.geometry.enable)
        self.assertFalse(k2.model.memory.enable)

        self.assertEqual(cached.model.training_attention_path, 'cached')
        self.assertFalse(cached.model.memory.enable)
        self.assertTrue(memory.model.memory.enable)
        self.assertTrue(memory.model.memory.action_read)
        self.assertFalse(cached.model.geometry.enable)
        self.assertTrue(geometry.model.geometry.enable)
        self.assertFalse(geometry.model.memory.enable)
        self.assertEqual(geometry.model.geometry.head.type, 'edge')
        for cfg in (baseline, k2, cached, memory, geometry):
            _validate_pipeline_config(cfg)

    def test_enhanced_config_preflight_rejects_late_shape_and_layer_errors(self):
        from endo4dwam.runtime import _validate_pipeline_config

        with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'configs')):
            source = compose(
                config_name='train',
                overrides=[
                    'task=endowam_fastwam_baseline_1cam_1e-4',
                    'history=k2',
                    'auxiliary=geometry',
                    'persistent_memory=on',
                ],
            )

        cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
        cfg.model.geometry.capture_layers = [12, 12]
        with self.assertRaisesRegex(ValueError, 'exactly four capture_layers'):
            _validate_pipeline_config(cfg)

        cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
        cfg.model.geometry.capture_layers = [12, 12, 16, 18]
        with self.assertRaisesRegex(ValueError, 'capture_layers must be unique'):
            _validate_pipeline_config(cfg)

        cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
        cfg.model.geometry.register_grid = [8, 9]
        with self.assertRaisesRegex(ValueError, 'register_grid.*video_size'):
            _validate_pipeline_config(cfg)

        cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
        cfg.model.geometry.num_supervised_steps = 3
        with self.assertRaisesRegex(ValueError, 'T_latent - K'):
            _validate_pipeline_config(cfg)

        cfg = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
        cfg.model.memory.capture_layer = 30
        with self.assertRaisesRegex(ValueError, 'memory.capture_layer'):
            _validate_pipeline_config(cfg)

    def test_rgb_is_cropped_not_squeezed(self):
        with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'configs')):
            cfg = compose(config_name='train', overrides=['task=endowam_uncond_1cam_1e-4'])
        for mode in ('train_transforms', 'val_transforms'):
            transforms = instantiate(cfg.data.train.processor[mode])
            x = torch.linspace(0,255,480).to(torch.uint8)[None,None,None,:].expand(1,3,360,480)
            for transform in transforms:
                x = transform(x)
            self.assertEqual(tuple(x.shape), (1,3,256,320))
            self.assertGreater(x[0,0,128,0].item(), .02)
            self.assertLess(x[0,0,128,-1].item(), .98)
        self.assertGreater(cfg.data.train.val_set_proportion, 0)
        self.assertFalse(cfg.data.val.is_training_set)

    def make_labels(self, folder, stride=2, legacy_qc=False, length=33):
        root = Path(folder) / 'ureter'
        g = root / 'geometry'
        for name in ('depth','depth_meta','flow','flow_meta','flow_mask'):
            (g/name).mkdir(parents=True)
        depth = np.broadcast_to(np.arange(length, dtype=np.float32)[:,None,None],(length,36,48)).copy()
        path = g/'depth/episode_000000.npy'
        np.save(path,depth)
        (g/'depth_meta/episode_000000.json').write_text(json.dumps({
            'num_frames':length,'height':36,'width':48,'model':'BaymaxShao/EdGE@unit-test',
            'teacher_mode':'causal_streaming','quality_status':'selected_after_video_ab_test'}))
        np.save(g/'flow/episode_000000.npy',np.broadcast_to(np.array([.1,-.2],dtype=np.float32)[None,:,None,None],(length,2,18,24)))
        np.save(g/'flow_mask/episode_000000.npy',np.ones((length,18,24),dtype=np.uint8))
        (g/'flow_meta/episode_000000.json').write_text(json.dumps({'num_frames':length,'stride':stride,'normalized_by_image_size':True,'flow_height':36,'flow_width':48,'store_height':18,'store_width':24}))
        qc = Path(folder)/'qc.json'
        qc.write_text(json.dumps([{'root':str(root),'episode':'episode_000000','passes':True,'num_frames':length,'depth_mtime_ns':path.stat().st_mtime_ns}]))
        datasets = [SimpleNamespace(root=root,episodes=[0],meta=SimpleNamespace(episodes={0:{'length':length}}))]
        cfg = {'num_history_latent_frames':1,'num_supervised_steps':4,'depth':True,'flow':True,
               'depth_qc_path':str(qc) if legacy_qc else None,'depth_procedure_weights':{'ureter':.4}}
        return datasets,cfg

    def test_labels_time_geometry_padding_and_signed_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            datasets,cfg = self.make_labels(tmp)
            reader=GeometryLabels(datasets,cfg,source_hw=(36,48),target_hw=(32,32),num_frames=33,video_stride=2,sample_stride=1)
            sample={'dataset_index':0,'episode_index':0,'frame_index':0}
            out=reader.read(sample)
            torch.testing.assert_close(out['depth'][:,10,10],torch.tensor([2.,10.,18.,26.]))
            self.assertEqual(tuple(out['flow'].shape),(4,2,32,32))
            torch.testing.assert_close(out['flow'][:,0],torch.full((4,32,32),.1*43/32))
            torch.testing.assert_close(out['flow'][:,1],torch.full((4,32,32),-.2))
            self.assertAlmostEqual(out['depth_weight'].item(),.4)
            sample['frame_index']=31
            out=reader.read(sample)
            self.assertEqual(out['flow_mask'].sum().item(),0)
            self.assertEqual(out['depth_mask'][1:].sum().item(),0)
            self.assertEqual(out['depth_mask'].sum().item(),0)

    def test_ambiguous_flow_fails_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            datasets,cfg=self.make_labels(tmp,stride=1)
            with self.assertRaisesRegex(ValueError,'\(2\)'):
                GeometryLabels(datasets,cfg,source_hw=(36,48),target_hw=(32,32),num_frames=33,video_stride=2,sample_stride=1)

    def test_video_stride_is_applied_to_label_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            datasets,cfg=self.make_labels(tmp)
            reader=GeometryLabels(datasets,cfg,source_hw=(36,48),target_hw=(32,32),num_frames=33,video_stride=2,sample_stride=1)
            out=reader.read({'dataset_index':0,'episode_index':0,'frame_index':0})
            torch.testing.assert_close(out['depth'][:,10,10],torch.tensor([2.,10.,18.,26.]))

    def test_missing_labels_and_stale_qc_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            datasets,cfg=self.make_labels(tmp,legacy_qc=True)
            path=datasets[0].root/'geometry/depth/episode_000000.npy'
            os.utime(path,ns=(path.stat().st_atime_ns,path.stat().st_mtime_ns+1))
            with self.assertRaisesRegex(ValueError,'Stale depth QC'):
                GeometryLabels(datasets,cfg,source_hw=(36,48),target_hw=(32,32),num_frames=33,video_stride=2,sample_stride=1)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                GeometryLabels(datasets,cfg,source_hw=(36,48),target_hw=(32,32),num_frames=33,video_stride=2,sample_stride=1)

    def test_k2_observed_flow_and_register_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            datasets,cfg=self.make_labels(tmp)
            cfg.update(num_history_latent_frames=2,num_supervised_steps=3,flow_include_observed=True)
            reader=GeometryLabels(datasets,cfg,source_hw=(36,48),target_hw=(32,32),num_frames=33,video_stride=2,sample_stride=1)
            out=reader.read({'dataset_index':0,'episode_index':0,'frame_index':0})
            torch.testing.assert_close(out['depth'][:,10,10],torch.tensor([10.,18.,26.]))
            self.assertEqual(out['flow'].shape[0],4)
        branch=build_geometry_branch(
            {'capture_layers':[0],'geo_dim':8,'num_heads':2,'depth':{'enable':True},
             'motion':{'enable':True,'include_observed':True}},
            video_dim=8,num_spatial=2,num_time=3,num_history=2)
        self.assertEqual(branch.depth.num_time,3)
        self.assertEqual(branch.motion.num_time,4)

    def test_fair_k2_window_keeps_four_future_geometry_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            datasets, cfg = self.make_labels(tmp, length=41)
            cfg.update(
                num_history_latent_frames=2,
                num_supervised_steps=4,
                flow_include_observed=False,
            )
            reader = GeometryLabels(
                datasets, cfg, source_hw=(36,48), target_hw=(32,32),
                num_frames=41, video_stride=2, sample_stride=1,
            )
            out = reader.read({'dataset_index':0, 'episode_index':0, 'frame_index':0})
            torch.testing.assert_close(out['depth'][:,10,10], torch.tensor([10.,18.,26.,34.]))
            self.assertEqual(tuple(out['flow'].shape), (4,2,32,32))

    def test_action_to_video_gradient_gate(self):
        x=torch.tensor([2.],requires_grad=True)
        y=scale_gradient(x,.05)
        self.assertEqual(y.item(),2.)
        y.backward()
        torch.testing.assert_close(x.grad,torch.tensor([.05]))

    def test_persistent_memory_is_explicit_visual_state(self):
        torch.manual_seed(7)
        memory=PersistentWorldMemory(
            video_dim=12,dim=8,num_tokens=4,num_heads=2,num_blocks=1,ffn_mult=2)
        initial=memory.initial_state(2,device='cpu',dtype=torch.float32)
        self.assertEqual(tuple(initial.shape),(2,4,8))
        self.assertEqual(initial.abs().sum().item(),0.)
        history=torch.randn(2,6,12,requires_grad=True)
        prior=torch.randn(2,4,8,requires_grad=True)
        state=memory.update(history,tokens_per_frame=3,state=prior,detach_previous=True)
        self.assertEqual(tuple(state.shape),(2,4,8))
        self.assertFalse(torch.equal(state,initial))
        state.square().mean().backward()
        self.assertIsNone(prior.grad)  # truncated across chunk boundaries
        self.assertGreater(history.grad.abs().sum().item(),0.)
        with self.assertRaisesRegex(ValueError,'divisible'):
            memory.update(history.detach(),tokens_per_frame=4)

    def test_geometry_registers_can_read_memory(self):
        torch.manual_seed(8)
        branch=build_geometry_branch(
            {'capture_layers':[0],'geo_dim':8,'num_heads':2,'depth':{'enable':True},
             'motion':{'enable':False}},
            video_dim=12,num_spatial=2,num_time=1,memory_dim=6)
        captures=[torch.randn(1,3,12)]
        zero=branch.depth(captures,memory=torch.zeros(1,4,6))[0]
        visual_memory=branch.depth(captures,memory=torch.randn(1,4,6))[0]
        self.assertFalse(torch.allclose(zero,visual_memory))

    def test_action_reads_memory_with_independent_gradient_gate(self):
        class SelfAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k=torch.nn.Linear(4,4,bias=False)
                self.v=torch.nn.Linear(4,4,bias=False)
                self.norm_k=torch.nn.Identity()
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.self_attn=SelfAttention()
        class Expert(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.blocks=torch.nn.ModuleList([Block()])
                self.num_heads=2; self.attn_head_dim=2
        mot=MoT({'video':Expert(),'action':Expert()},mot_checkpoint_mixed_attn=False)
        mot.world_memory=PersistentWorldMemory(
            video_dim=4,dim=4,num_tokens=2,num_heads=2,num_blocks=1,ffn_mult=2)

        def build_io(_self,expert,block,x,freqs,t_mod):
            z=torch.zeros_like(x)
            return x,x,x,x,z,z,z,z,False
        seen=[]
        def mixed(_self,q_cat,k_cat,v_cat,attention_mask):
            seen.append(tuple(attention_mask.shape))
            scores=torch.matmul(q_cat,k_cat.transpose(1,2))/2.
            scores=scores.masked_fill(~attention_mask.unsqueeze(0),-1e4)
            return torch.matmul(scores.softmax(dim=-1),v_cat)
        def post(_self,**kwargs): return kwargs['mixed_slice']
        mot._build_expert_attention_io=MethodType(build_io,mot)
        mot._mixed_attention=MethodType(mixed,mot)
        mot._apply_post_with_optional_checkpoint=MethodType(post,mot)

        state=torch.randn(1,2,4,requires_grad=True)
        out=mot.forward_action_with_video_cache(
            action_tokens=torch.randn(1,2,4),action_freqs=torch.empty(0),
            action_t_mod=torch.empty(0),action_context_payload=None,
            video_kv_cache=[{'k':torch.randn(1,3,4),'v':torch.randn(1,3,4)}],
            attention_mask=torch.ones(5,5,dtype=torch.bool),video_seq_len=3,
            memory_state=state,action_to_memory_grad_scale=0.)
        out.sum().backward()
        self.assertEqual(seen[-1],(2,7))  # 3 video + 2 memory + 2 action keys
        self.assertEqual(state.grad.abs().sum().item(),0.)

    def test_cached_no_memory_matches_mixed_attention_forward(self):
        class SelfAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = self.k = self.v = self.o = torch.nn.Identity()
                self.norm_q = self.norm_k = torch.nn.Identity()
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = SelfAttention()
        class Expert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block()])
                self.num_heads = 2
                self.attn_head_dim = 2
        mot = MoT({'video': Expert(), 'action': Expert()}, mot_checkpoint_mixed_attn=False)

        def build_io(_self, expert, block, x, freqs, t_mod):
            z = torch.zeros_like(x)
            return x, x, x, x, z, z, z, z, False
        def mixed(_self, q_cat, k_cat, v_cat, attention_mask):
            scores = torch.matmul(q_cat, k_cat.transpose(1, 2)) / 2.
            scores = scores.masked_fill(~attention_mask.unsqueeze(0), -1e4)
            return torch.matmul(scores.softmax(dim=-1), v_cat)
        def post(_self, **kwargs):
            return kwargs['mixed_slice']
        mot._build_expert_attention_io = MethodType(build_io, mot)
        mot._mixed_attention = MethodType(mixed, mot)
        mot._apply_post_with_optional_checkpoint = MethodType(post, mot)

        video = torch.randn(1, 3, 4)
        action = torch.randn(1, 2, 4)
        mask = torch.zeros(5, 5, dtype=torch.bool)
        mask[:3, :3] = True
        mask[3:, 0] = True
        mask[3:, 3:] = True
        empty = torch.empty(0)
        joint = mot(
            embeds_all={'video': video, 'action': action},
            attention_mask=mask,
            freqs_all={'video': empty, 'action': empty},
            context_all={'video': None, 'action': None},
            t_mod_all={'video': empty, 'action': empty},
        )
        cache, cached_video = mot.prefill_video_cache(
            video_tokens=video,
            video_freqs=empty,
            video_t_mod=empty,
            video_context_payload=None,
            video_attention_mask=mask[:3, :3],
            return_final_tokens=True,
        )
        cached_action = mot.forward_action_with_video_cache(
            action_tokens=action,
            action_freqs=empty,
            action_t_mod=empty,
            action_context_payload=None,
            video_kv_cache=cache,
            attention_mask=mask,
            video_seq_len=3,
        )
        torch.testing.assert_close(cached_video, joint['video'])
        torch.testing.assert_close(cached_action, joint['action'])

    def test_trainer_preserves_head_freeze(self):
        from endo4dwam.trainer import Wan22Trainer
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dit=torch.nn.Module()
                self.dit.geometry=torch.nn.Module()
                self.dit.geometry.depth_head=torch.nn.Linear(2,2)
                self.dit.geometry.register=torch.nn.Linear(2,2)
                self.geometry_config={'head':{'freeze':True}}
            @property
            def geometry(self):
                return self.dit.geometry
        model=Model()
        Wan22Trainer._apply_dit_only_train_mode(model)
        self.assertFalse(model.geometry.depth_head.training)
        self.assertTrue(all(not p.requires_grad for p in model.geometry.depth_head.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.geometry.register.parameters()))

    def test_action_only_eval_logging_does_not_require_video_metrics(self):
        from endo4dwam.trainer import Wan22Trainer
        trainer = Wan22Trainer.__new__(Wan22Trainer)
        trainer.global_step = 500
        description, payload = trainer._format_eval_metrics({
            "val_loss": 1.25,
            "action_l2": 0.5,
            "action_l1": 0.25,
        })
        self.assertIn("val_loss=1.2500", description)
        self.assertIn("action_l1=0.2500", description)
        self.assertNotIn("infer_psnr", description)
        self.assertEqual(payload, {
            "eval/val_loss": 1.25,
            "eval/action_l2": 0.5,
            "eval/action_l1": 0.25,
        })

    def test_accelerator_without_deepspeed_is_supported(self):
        from endo4dwam.trainer import Wan22Trainer
        accelerator = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None))
        self.assertEqual(Wan22Trainer._zero_stage(accelerator), "none")
        accelerator.state.deepspeed_plugin = SimpleNamespace(
            deepspeed_config={"zero_optimization": {"stage": 2}}
        )
        self.assertEqual(Wan22Trainer._zero_stage(accelerator), 2)

    def test_curriculum_progressively_unfreezes_real_optimizer_params(self):
        from endo4dwam.trainer import Wan22Trainer
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mot=torch.nn.Module()
                self.mot.mixtures=torch.nn.ModuleDict({
                    'video':torch.nn.Linear(2,2), 'action':torch.nn.Linear(2,2)})
                self.mot.geometry=torch.nn.Module()
                self.mot.geometry.register=torch.nn.Linear(2,2)
                self.mot.geometry.depth_head=torch.nn.Linear(2,2)
                self.mot.world_memory=torch.nn.Linear(2,2)
                self.dit=self.mot
                self.proprio_encoder=torch.nn.Linear(2,2)
                self.geometry_config={'head':{'freeze':False}}
                self.action_to_video_grad_scale=1.
                self._lora_train_base=True
            @property
            def geometry(self): return self.mot.geometry
            @property
            def world_memory(self): return self.mot.world_memory
        trainer=Wan22Trainer.__new__(Wan22Trainer)
        trainer.curriculum={
            'enable':True,'geometry_unfreeze_step':0,'head_unfreeze_step':0,
            'memory_unfreeze_step':0,
            'video_unfreeze_step':5,'action_unfreeze_step':10,
            'action_to_video_grad_scale_before':0.,'action_to_video_grad_scale_after':.05,
            'action_to_memory_grad_scale_before':0.,'action_to_memory_grad_scale_after':.05}
        model=Model()
        Wan22Trainer._apply_dit_only_train_mode(model)
        trainer._apply_curriculum(model,0)
        self.assertFalse(any(p.requires_grad for p in model.mot.mixtures['video'].parameters()))
        self.assertFalse(any(p.requires_grad for p in model.mot.mixtures['action'].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.world_memory.parameters()))
        ids={id(p) for p in trainer._optimizer_parameters(model)}
        self.assertTrue(all(id(p) in ids for p in model.mot.mixtures['action'].parameters()))
        trainer._apply_curriculum(model,10)
        self.assertTrue(all(p.requires_grad for p in model.mot.mixtures['video'].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.mot.mixtures['action'].parameters()))
        self.assertEqual(model.action_to_video_grad_scale,.05)
        self.assertEqual(model.action_to_memory_grad_scale,.05)

    def test_depth_procedure_weight_does_not_cancel(self):
        torch.manual_seed(4)
        pred=torch.randn(1,4,4,4,requires_grad=True)
        target=torch.randn_like(pred)
        loss,*_=clip_shared_affine_depth_loss(pred,target)
        weighted,*_=clip_shared_affine_depth_loss(pred,target,sample_weight=torch.tensor([.4]))
        torch.testing.assert_close(weighted,loss*.4)
        weighted.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        flow=torch.randn(1,4,2,4,4,requires_grad=True)
        zero=flow_loss(flow,torch.zeros_like(flow),torch.zeros(1,4,4,4))
        zero.backward()
        self.assertEqual(zero.item(),0)
        self.assertEqual(flow.grad.abs().sum().item(),0)
        exact=flow_loss(torch.ones(1,1,2,2,2),torch.ones(1,1,2,2,2))
        self.assertEqual(exact.item(),0.)

    def test_joint_gets_video_length(self):
        class Joint:
            device='cpu'
            torch_dtype=torch.float32
            def infer_action(self, *, num_video_frames, **kwargs):
                return num_video_frames
        batched={'video':torch.zeros(1,3,17,32,32),'proprio':torch.zeros(1,32,3),'context':torch.zeros(1,2,4),'context_mask':torch.ones(1,2)}
        args=SimpleNamespace(text_cfg_scale=1.,num_inference_steps=1,seed=0)
        self.assertEqual(infer_action_window(Joint(),batched,args,32),17)

    def test_chunk_eval_passes_explicit_memory_state(self):
        marker=torch.randn(1,2,3)
        class Base:
            device='cpu'; torch_dtype=torch.float32
            def infer_action(self, *, memory_state=None, **kwargs):
                return {'action':torch.zeros(2,3),'memory_state':memory_state}
        batched={'video':torch.zeros(1,3,17,32,32),'proprio':torch.zeros(1,32,3),
                 'context':torch.zeros(1,2,4),'context_mask':torch.ones(1,2)}
        args=SimpleNamespace(text_cfg_scale=1.,num_inference_steps=1,seed=0)
        out=infer_action_window(Base(),batched,args,32,memory_state=marker)
        self.assertIs(out['memory_state'],marker)

    def test_eval_uses_saved_alpha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ckpt=root/'checkpoints/weights/step_1.pt'; ckpt.parent.mkdir(parents=True)
            OmegaConf.save({'model':{'lora':{'alpha':77}}}, root/'config.yaml')
            self.assertEqual(load_training_config(ckpt).model.lora.alpha,77)

    def test_launch_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'bin').mkdir()
            executable=root/'bin/accelerate'; executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n'); executable.chmod(0o755)
            env=dict(os.environ,PATH=str(root/'bin')+':'+os.environ['PATH'],RUN_ID='review',RUN_ROOT=str(root/'runs'),NNODES='2',NODE_RANK='1',MASTER_ADDR='192.0.2.1',MASTER_PORT='29999')
            for zero in (1,2):
                r=subprocess.run(['bash',str(ROOT/f'scripts/train_zero{zero}.sh'),'2','task=endowam_uncond_1cam_1e-4'],env=env,capture_output=True,text=True)
                self.assertEqual(r.returncode,0,r.stderr)
                self.assertIn('--num_processes\n4\n',r.stdout)
                self.assertIn('--num_machines\n2\n',r.stdout)
                self.assertIn('--machine_rank\n1\n',r.stdout)
                self.assertIn('--main_process_port\n29999\n',r.stdout)
            state=root/'runs/review/checkpoints/state'; state.mkdir(parents=True)
            for name in ('uncond','joint'):
                script=ROOT/f'scripts/train_endowam_lora_{name}.sh'
                r=subprocess.run(['bash',str(script),'--resume'],env=env,capture_output=True,text=True)
                self.assertNotEqual(r.returncode,0)
                self.assertIn('No complete training state',r.stderr)
            for step in (9,100):
                d=state/f'step_{step:06d}'; d.mkdir(); (d/'trainer_state.json').write_text('{}')
            r=subprocess.run(['bash',str(ROOT/'scripts/train_endowam_lora_uncond.sh'),'--resume','batch_size=1'],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('step_000100',r.stdout)
            self.assertIn('batch_size=1',r.stdout)


if __name__=='__main__':
    unittest.main()
