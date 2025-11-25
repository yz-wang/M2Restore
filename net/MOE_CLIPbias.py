import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.normal import Normal
import numpy as np

class ES_EE(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
    def es(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)
    def ee(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        ensemble = zeros.index_add(0, self._batch_index, stitched.float())
        ensemble[ensemble == 0] = np.finfo(float).eps
        return ensemble.log()
    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class Mlp(nn.Module):
    def __init__(self, in_feat, h_feat=None, out_feat=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_feat = out_feat or in_feat
        h_feat = h_feat or in_feat
        self.fc1 = nn.Linear(in_feat, h_feat)
        self.act = act_layer()
        self.fc2 = nn.Linear(h_feat, out_feat)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
        return x

class model(nn.Module):
    def __init__(self, input_size, output_size, mlp_ratio, num_experts, no=True, use_experts=2, num_degradations=6):
        super(model, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size

        self.degra_proj = nn.Linear(num_degradations, num_experts)
        self.gate_boost = nn.Parameter(torch.tensor(0.5))

        self.el = nn.ModuleList([Mlp(input_size, h_feat=int(input_size * mlp_ratio), out_feat=output_size) for i in range(self.num_experts)])

        self.input_size = input_size
        self.no = no
        self.softmax = nn.Softmax(1)
        self.k = use_experts
        self.w_g = nn.Parameter(torch.randn(2 * input_size, num_experts), requires_grad=True)
        self.sp = nn.Softplus()
        self.w_n = nn.Parameter(torch.zeros(2 * input_size, num_experts), requires_grad=True)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.test_activations = []
        assert (self.k <= self.num_experts)
    def balance(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)
    def noisyk_w(self, x, de_cls, train, noise_epsilon=1e-2):
        expert_bias = self.degra_proj(de_cls)  # (B*H*W, num_experts)
        clean_logits = x @ self.w_g
        boosted_logits = clean_logits + self.gate_boost * expert_bias
        if self.no and train:
            raw_noise_stddev = x @ self.w_n
            noise_stddev = ((self.sp(raw_noise_stddev) + noise_epsilon))
            noisy_logits = boosted_logits + (torch.randn_like(boosted_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = boosted_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        w = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.no and self.k < self.num_experts and train:
            n = (self.index_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            n = self.w_load(w)
        return w, n
    def w_load(self, gates):
        return (gates > 0).sum(0)
    def index_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_val = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_val, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_val, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def forward(self, x, prompt, de_cls):
        B, C, H, W = x.shape
        prompt = prompt.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x = rearrange(x, 'b c h w -> (b h w) c')
        prompt = rearrange(prompt, 'b c h w -> (b h w) c')
        x_p = torch.cat((x, prompt), dim=1)

        de_cls_expanded = de_cls.view(B, -1, 1, 1).expand(-1, -1, H, W)
        de_cls_flat = rearrange(de_cls_expanded, 'b d h w -> (b h w) d')

        w, s = self.noisyk_w(x_p, de_cls_flat, self.training)

        Weight = w.sum(0)
        loss_Weight = self.balance(Weight)
        loss_Sum = self.balance(s)
        loss = loss_Weight + loss_Sum

        es_ee = ES_EE(self.num_experts, w)
        expert_inputs = es_ee.es(x)
        gates = es_ee.expert_to_gates()
        expert_outputs = [self.el[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = es_ee.ee(expert_outputs)
        y = rearrange(y, '(b h w) c -> b c h w', b=B, h=H, w=W)

        if not self.training:
            self.test_activations.append(w.detach().cpu())
        return y, loss

class MCDB(nn.Module):
    def __init__(self, atom_dim, dim, ffn_expansion_factor):
        super(MCDB, self).__init__()
        self.fc = nn.Linear(atom_dim, dim)
        self.model = model(dim, dim, mlp_ratio=ffn_expansion_factor, num_experts=6, no=True, use_experts=2)
    def forward(self, x, prompt, de_cls):
        d = self.fc(prompt)
        out, loss = self.model(x, d, de_cls)
        return out + x, loss