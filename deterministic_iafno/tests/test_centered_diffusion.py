# 用途：验证冻结均值和中心化扩散的训练采样行为。
import unittest

import torch
from torch import nn

from deterministic_iafno.centered_diffusion import (
    FrozenMeanCenteredDiffusion,
)


class RecordingMean(nn.Module):
    """Deterministic mean stub whose predict() returns a constant
    per-lead normalized residual."""

    # 用途：测试桩的初始化。
    # 参数：见签名；输出 无。
    def __init__(self, values=(0.0, 0.0)):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(
            values, dtype=torch.float32
        ), requires_grad=False)
        self.train_calls = 0

    # 用途：执行短训练的测试辅助。
    # 参数：见签名；输出 无。
    def train(self, mode=True):
        super().train(mode)
        self.train_calls += 1
        return self

    # 用途：执行一次预测的测试辅助。
    # 参数：见签名；输出 预测结果。
    def predict(self, condition):
        shape = (
            condition.shape[0],
            self.values.numel(),
            *condition.shape[2:],
        )
        return self.values.view(
            1, -1, 1, 1, 1
        ).expand(shape).clone()


class RecordingDiffusion(nn.Module):
    """Diffusion stub recording the standardized target it receives."""

    # 用途：测试桩的初始化。
    # 参数：见签名；输出 无。
    def __init__(self):
        super().__init__()
        self.channels = 2
        self.linear = nn.Linear(4, 2)
        self.S_churn = 0.0
        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.rho = 7.0
        self.num_sample_steps = 16
        self.last_target = None
        self.last_mask = None
        self.sample_calls = 0
        self.last_num_steps = None
        self.last_seed = None
        self.sample_output = None

    # 用途：执行一次前向的测试辅助。
    # 参数：见签名；输出 模型输出。
    def forward(self, target, condition, target_mask=None):
        self.last_target = target.detach().clone()
        self.last_mask = (
            None
            if target_mask is None
            else target_mask.detach().clone()
        )
        weight_sum = (
            self.linear.weight.sum() + self.linear.bias.sum()
        )
        return (target * target).mean() + weight_sum

    @torch.no_grad()
    # 用途：执行一次采样的测试辅助。
    # 参数：见签名；输出 采样结果。
    def sample(self, condition, num_sample_steps=None, seed=None):
        self.sample_calls += 1
        self.last_num_steps = num_sample_steps
        self.last_seed = seed
        if self.sample_output is not None:
            return self.sample_output.clone()
        return torch.zeros(
            condition.shape[0],
            self.channels,
            *condition.shape[2:],
            device=condition.device,
            dtype=torch.float32,
        )


# 用途：构建测试用 centered 包装器的辅助函数。
# 参数：见签名；输出 包装器实例。
def build_wrapper(innovation_mean=(0.0, 0.0), innovation_std=(1.0, 1.0),
                  mean_values=(0.0, 0.0), diffusion=None):
    mean = RecordingMean(mean_values)
    diffusion = (
        RecordingDiffusion() if diffusion is None else diffusion
    )
    return FrozenMeanCenteredDiffusion(
        mean,
        diffusion,
        lead_mean=innovation_mean,
        lead_std=innovation_std,
    ), mean, diffusion


# 用途：构造测试用条件张量的辅助函数。
# 参数：见签名；输出 条件张量。
def condition(batch=2):
    return torch.randn(batch, 8, 2, 2, 1)


class CenteredAlgebraTests(unittest.TestCase):
    # 用途：验证 forward 计算的是标准化 innovation。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_forward_computes_standardized_innovation(self):
        wrapper, mean, diffusion = build_wrapper(
            innovation_mean=(1.0, -1.0),
            innovation_std=(2.0, 4.0),
            mean_values=(0.5, -0.5),
        )
        cond = condition()
        residual_target = torch.randn(2, 2, 2, 2, 1)
        wrapper(residual_target, cond, None)
        mu = torch.tensor([0.5, -0.5]).view(1, 2, 1, 1, 1)
        expected = (
            (residual_target - mu)
            - torch.tensor([1.0, -1.0]).view(1, 2, 1, 1, 1)
        ) / torch.tensor([2.0, 4.0]).view(1, 2, 1, 1, 1)
        self.assertTrue(torch.allclose(
            diffusion.last_target, expected, atol=1e-6
        ))

    # 用途：验证 forward 把 target_mask 透传给内部扩散。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_forward_passes_target_mask_through(self):
        wrapper, mean, diffusion = build_wrapper()
        cond = condition()
        mask = torch.ones(2, 2, 2, 2, 1)
        wrapper(torch.randn(2, 2, 2, 2, 1), cond, mask)
        self.assertIsNotNone(diffusion.last_mask)
        self.assertTrue(torch.equal(
            diffusion.last_mask, mask
        ))

    # 用途：验证 AMP 下标准化仍保持 fp32。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_standardization_stays_fp32_under_amp(self):
        wrapper, mean, diffusion = build_wrapper()
        cond = condition()
        with torch.autocast("cpu", enabled=True):
            wrapper(torch.randn(2, 2, 2, 2, 1), cond, None)
        self.assertEqual(
            diffusion.last_target.dtype, torch.float32
        )

    # 用途：验证形状不一致时失败。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_shape_mismatch_fails(self):
        wrapper, mean, diffusion = build_wrapper()
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            wrapper(
                torch.randn(3, 2, 2, 2, 1),
                condition(batch=2),
                None,
            )
        with self.assertRaisesRegex(ValueError, "spatial shapes"):
            wrapper(
                torch.randn(2, 2, 4, 4, 1),
                condition(batch=2),
                None,
            )


class CenteredFreezeTests(unittest.TestCase):
    # 用途：验证均值冻结且扩散分支可训练。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_mean_frozen_and_diffusion_trainable(self):
        wrapper, mean, diffusion = build_wrapper()
        for parameter in mean.parameters():
            self.assertFalse(parameter.requires_grad)
        trainable = [
            parameter
            for parameter in wrapper.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(
            parameter is not mean.values for parameter in trainable
        ))

    # 用途：验证 train() 下均值模型保持 eval。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_train_mode_keeps_mean_eval(self):
        wrapper, mean, diffusion = build_wrapper()
        self.assertFalse(mean.training)
        wrapper.train()
        self.assertTrue(diffusion.linear.training)
        self.assertFalse(mean.training)
        wrapper.eval()
        wrapper.train()
        self.assertFalse(mean.training)

    # 用途：验证反向传播只更新扩散分支、均值模型梯度为零。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_backward_updates_diffusion_not_mean(self):
        wrapper, mean, diffusion = build_wrapper()
        cond = condition()
        loss = wrapper(torch.randn(2, 2, 2, 2, 1), cond, None)
        loss.backward()
        self.assertIsNone(mean.values.grad)
        self.assertIsNotNone(diffusion.linear.weight.grad)

    # 用途：验证优化器参数列表排除 mean_model。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_optimizer_params_exclude_mean(self):
        wrapper, mean, diffusion = build_wrapper()
        optimizer = torch.optim.AdamW(
            [
                parameter
                for parameter in wrapper.parameters()
                if parameter.requires_grad
            ],
            lr=1e-3,
        )
        optimizer.step()
        self.assertEqual(len(optimizer.param_groups[0]["params"]), 2)
        self.assertTrue(all(
            parameter is not mean.values
            for parameter in optimizer.param_groups[0]["params"]
        ))


class CenteredSamplingTests(unittest.TestCase):
    # 用途：验证标准化-反变换往返。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_transform_inverse_round_trip(self):
        wrapper, mean, diffusion = build_wrapper(
            innovation_mean=(0.3, -0.2),
            innovation_std=(1.5, 2.5),
        )
        innovation = torch.randn(2, 2, 2, 2, 1)
        restored = wrapper.inverse_innovation(
            wrapper.transform_innovation(innovation)
        )
        self.assertTrue(torch.allclose(
            innovation, restored, atol=1e-6
        ))

    # 用途：验证采样返回不含 anchor 的 normalized residual。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_sample_returns_normalized_residual_no_anchor(self):
        # mean=(mu), z_hat=0 -> r_hat = mu + m (+0).  The wrapper must
        # NOT add the day-7 anchor and must NOT denormalize to SST.
        wrapper, mean, diffusion = build_wrapper(
            innovation_mean=(0.25, -0.5),
            innovation_std=(1.0, 2.0),
            mean_values=(0.5, 0.5),
        )
        cond = condition(batch=1)
        forecast = wrapper.sample(cond, num_sample_steps=4, seed=7)
        self.assertEqual(diffusion.sample_calls, 1)
        self.assertEqual(diffusion.last_num_steps, 4)
        self.assertEqual(diffusion.last_seed, 7)
        self.assertEqual(tuple(forecast.shape), (1, 2, 2, 2, 1))
        self.assertEqual(forecast.dtype, torch.float32)
        expected = torch.tensor(
            [[0.75, 0.0]]
        ).view(1, 2, 1, 1, 1).expand(1, 2, 2, 2, 1)
        self.assertTrue(torch.allclose(forecast, expected, atol=1e-6))

    # 用途：验证零 innovation 时退化为确定性均值。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_zero_innovation_degenerates_to_deterministic_mean(self):
        # With zero innovation stats (m=0, s=1), a zero z_hat sample
        # must equal the deterministic mean exactly.
        wrapper, mean, diffusion = build_wrapper(
            innovation_mean=(0.0, 0.0),
            innovation_std=(1.0, 1.0),
            mean_values=(1.5, -2.0),
        )
        cond = condition(batch=1)
        forecast = wrapper.sample(cond, seed=0)
        mu = torch.tensor([1.5, -2.0]).view(
            1, 2, 1, 1, 1
        ).expand(1, 2, 2, 2, 1)
        self.assertTrue(torch.allclose(forecast, mu, atol=1e-6))

    # 用途：验证采样器属性的双向透传。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_sampler_attributes_two_way_delegation(self):
        wrapper, mean, diffusion = build_wrapper()
        self.assertEqual(wrapper.S_churn, diffusion.S_churn)
        wrapper.S_churn = 0.25
        self.assertEqual(diffusion.S_churn, 0.25)
        self.assertEqual(wrapper.S_churn, 0.25)

        wrapper.sigma_min = 0.001
        self.assertEqual(diffusion.sigma_min, 0.001)
        self.assertEqual(wrapper.sigma_min, 0.001)

        wrapper.sigma_max = 90.0
        self.assertEqual(diffusion.sigma_max, 90.0)
        self.assertEqual(wrapper.sigma_max, 90.0)

        wrapper.rho = 6.0
        self.assertEqual(diffusion.rho, 6.0)
        self.assertEqual(wrapper.rho, 6.0)

        wrapper.num_sample_steps = 32
        self.assertEqual(diffusion.num_sample_steps, 32)
        self.assertEqual(wrapper.num_sample_steps, 32)

    # 用途：验证非法 innovation 统计被拒绝。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_invalid_innovation_stats_fail(self):
        with self.assertRaisesRegex(ValueError, "channels"):
            FrozenMeanCenteredDiffusion(
                RecordingMean(),
                RecordingDiffusion(),
                lead_mean=[0.0],
                lead_std=[1.0, 1.0],
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            FrozenMeanCenteredDiffusion(
                RecordingMean(),
                RecordingDiffusion(),
                lead_mean=[0.0, 0.0],
                lead_std=[1.0, 0.0],
            )

    # 用途：验证包装器不暴露 preconditioned_network_forward。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_no_preconditioned_network_forward_exposed(self):
        wrapper, mean, diffusion = build_wrapper()
        self.assertFalse(hasattr(
            wrapper, "preconditioned_network_forward"
        ))


class CenteredTinyBackboneTests(unittest.TestCase):
    """CPU forward/backward/sample through the real IAFNODiff trunks."""

    # 用途：构建测试用对象的辅助函数。
    # 参数：见签名；输出 测试用对象。
    def _build(self):
        from diafno.models.config import OSTIAModelConfig
        config = OSTIAModelConfig(
            image_size=(8, 8, 1),
            patch_size=(2, 2, 1),
            embed_dim=8,
            num_blocks=2,
            explicit_layer=1,
            implicit_layer=1,
            hidden_size_factor=2,
            cond_chans=3,
            target_chans=2,
            sampling_steps=4,
            sigma_data=1.0,
            sigma_max=10.0,
            sigma_min=0.002,
            target_mode="residual",
            model_type="centered_diffusion",
            target_scaling="lead_standardized",
            lead_mean=(0.1, -0.1),
            lead_std=(1.0, 2.0),
            mean_lead_mean=(0.0, 0.1),
            mean_lead_std=(1.0, 1.5),
            mean_checkpoint_sha256="a" * 64,
            mean_semantics_sha256="b" * 64,
        )
        model = config.build_model(torch.device("cpu"))
        return config, model

    # 用途：验证前向反向的损失与梯度均有限。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_forward_backward_finite(self):
        config, model = self._build()
        cond = torch.randn(2, 3, 8, 8, 1)
        residual = torch.randn(2, 2, 8, 8, 1)
        mask = torch.ones_like(residual)
        loss = model(residual, cond, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        diffusion_grads = [
            parameter.grad
            for parameter in model.diffusion.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(diffusion_grads)
        self.assertTrue(any(
            torch.count_nonzero(gradient) > 0
            for gradient in diffusion_grads
        ))
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.mean_model.parameters()
        ))

    # 用途：验证采样输出为有限的 normalized residual。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_sample_finite_normalized_residual(self):
        config, model = self._build()
        cond = torch.randn(1, 3, 8, 8, 1)
        forecast = model.sample(cond, num_sample_steps=4, seed=3)
        self.assertEqual(tuple(forecast.shape), (1, 2, 8, 8, 1))
        self.assertTrue(torch.isfinite(forecast).all())

    # 用途：验证 state_dict 包含冻结均值权重。
    # 参数：无输入（unittest 夹具自建合成数据）；输出 无（断言失败即抛异常）。
    def test_state_dict_contains_frozen_mean(self):
        config, model = self._build()
        keys = set(model.state_dict().keys())
        self.assertTrue(any(
            key.startswith("mean_model.") for key in keys
        ))
        self.assertTrue(any(
            key.startswith("diffusion.") for key in keys
        ))
        self.assertNotIn("innovation_mean", keys)
        self.assertNotIn("innovation_std", keys)


if __name__ == "__main__":
    unittest.main()
