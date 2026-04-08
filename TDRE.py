import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# ==========================================
# stage 1: Degradation Perception

class ClearSkyPerceptron(nn.Module):
    def __init__(self, requires_grad=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # B,8,1,1
        self.fc_block = nn.Sequential(
            nn.Linear(8, 4),  # 8→4
            nn.ReLU(inplace=True),
            nn.Linear(4, 1)  # 4→1
        )
        self.conv[0].weight.requires_grad = requires_grad
        self.conv[1].weight.requires_grad = requires_grad
        self.conv[1].bias.requires_grad = requires_grad

        self.fc_block[0].weight.requires_grad = requires_grad
        self.fc_block[2].weight.requires_grad = requires_grad
        self.fc_block[0].bias.requires_grad = requires_grad
        self.fc_block[2].bias.requires_grad = requires_grad

    def freeze_bn(self):

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).view(x.size(0), 8)
        return self.fc_block(x).squeeze(1)



class SharedTrunk(nn.Module):
    def __init__(self, inter_ch=8, requires_grad=True):
        super().__init__()
        self.c1 = nn.Conv2d(3, inter_ch, 1, 1, 0, bias=True)
        self.c2 = nn.Conv2d(inter_ch, inter_ch, 3, 1, 1, bias=True)
        self.c1.weight.requires_grad = requires_grad
        self.c2.weight.requires_grad = requires_grad
        self.c2.bias.requires_grad = requires_grad
        self.c1.bias.requires_grad = requires_grad

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        return x


class MultiScaleExpert(nn.Module):
    def __init__(self, inter_ch=8, requires_grad=True):
        super().__init__()
        self.c3 = nn.Conv2d(inter_ch, inter_ch, 5, 1, 2, bias=True)
        self.c4 = nn.Conv2d(inter_ch, inter_ch, 7, 1, 3, bias=True)
        self.c5 = nn.Conv2d(inter_ch, inter_ch, 3, 1, 1, bias=True)

        self.c3.weight.requires_grad = requires_grad
        self.c4.weight.requires_grad = requires_grad
        self.c5.weight.requires_grad = requires_grad

        self.c3.bias.requires_grad = requires_grad
        self.c4.bias.requires_grad = requires_grad
        self.c5.bias.requires_grad = requires_grad

    def forward(self, f):
        f3 = F.relu(self.c3(f))
        f4 = F.relu(self.c4(f3))
        f5 = F.relu(self.c5(f4))
        return torch.cat([f3, f4, f5], dim=1)  # 3*inter_ch


# ==========================================
# stage 2: Condition-Specific Restoration

class DRMoE(nn.Module):
    def __init__(self, n_expert=3, top_k=1, inter_ch=3, requires_grad=True):
        super().__init__()
        self.n_expert = n_expert
        self.top_k = top_k
        self.inter_ch = inter_ch

        self.trunk = SharedTrunk(inter_ch, requires_grad=requires_grad)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(inter_ch, n_expert)
        )
        self.experts = nn.ModuleList([
            MultiScaleExpert(inter_ch, requires_grad=requires_grad) for _ in range(n_expert)
        ])
        self.fusion = nn.Conv2d(3 * inter_ch, 3, 1, 1, 0, bias=True)

        self.gate[2].weight.requires_grad = requires_grad
        self.gate[2].bias.requires_grad = requires_grad
        self.fusion.weight.requires_grad = requires_grad
        self.fusion.bias.requires_grad = requires_grad

    def forward(self, x):
        f2 = self.trunk(x)
        logits = self.gate(f2)
        g = F.softmax(logits, dim=1)

        if self.top_k == 1:
            idx = torch.argmax(g, 1)
            one_hot = F.one_hot(idx, self.n_expert).float()
            g = g * one_hot
        else:
            topk_val, topk_idx = torch.topk(g, self.top_k, dim=1)
            g = torch.zeros_like(g).scatter_(1, topk_idx, topk_val)
            g = g / (g.sum(1, keepdim=True) + 1e-8)

        expert_outs = []
        for i, expert in enumerate(self.experts):
            if g[:, i].sum() == 0:
                dummy = torch.zeros_like(f2).repeat(1, 3, 1, 1)
                expert_outs.append(dummy)
                continue
            expert_outs.append(expert(f2))

        weighted = sum(g[:, i:i + 1, None, None] * expert_outs[i]
                       for i in range(self.n_expert))
        out = self.fusion(weighted)

        clean = F.relu(out * x - out + 1)

        return clean, logits


# ==========================================
# stage 3: Detection-Oriented Enhancement

class DetectionEnhancement(nn.Module):
    def __init__(self, kernel_size=3, act=nn.ReLU(inplace=True)):
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size, padding=pad, bias=True),
            act,
            nn.Conv2d(8, 3, 1, bias=True)
        )

    def forward(self, x):
        return self.body(x) + x



class TDRE(nn.Module):
    def __init__(self, n_expert=3, top_k=3, inter_ch=3, requires_grad=True):
        super().__init__()

        self.clf = ClearSkyPerceptron(requires_grad=False)

        self.moe = DRMoE(n_expert, top_k, inter_ch, requires_grad)

        self.enhancer = DetectionEnhancement(kernel_size=3)

    @torch.no_grad()
    def set_bn_to_eval(self):
        self.clf.freeze_bn()

    def _is_clear(self, x):
        logit = self.clf(x)
        return torch.sigmoid(logit) > 0.5

    def forward(self, x):
        clear_mask = self._is_clear(x)  # N

        clean = torch.zeros_like(x)
        logits_moe = torch.zeros(x.size(0), self.moe.n_expert, device=x.device)
        enhance = torch.zeros_like(x)

        # ---- Clear Branch ----
        if clear_mask.any():
            clean[clear_mask] = x[clear_mask]
            enhance[clear_mask] = clean[clear_mask]

        # ---- Degraded Branch ---
        if (~clear_mask).any():
            x_blur = x[~clear_mask]


            clean_blur, logits_blur = self.moe(x_blur)
            clean[~clear_mask] = clean_blur
            logits_moe[~clear_mask] = logits_blur


            enhance_blur = self.enhancer(clean_blur)
            enhance[~clear_mask] = enhance_blur

        return clean, logits_moe, enhance


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


if __name__ == "__main__":
    B, C, H, W = 1, 3, 256, 256
    x = torch.randn(B, C, H, W)

    model = TDRE(n_expert=3, top_k=3, inter_ch=3)
    model_structure(model)

    flops, params = profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops:,}")
    print(f"MACs (≈FLOPs/2): {flops / 2:,.0f}")

    model.eval()
    with torch.no_grad():
        clean, logits, enhance = model(x)
        print("\nInput shape :", x.shape)
        print("Restored (Clean) shape:", clean.shape)
        print("Enhanced shape:", enhance.shape)
        print("Expert Logits:", logits)