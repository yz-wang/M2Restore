import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

from mamba_ssm import Mamba
from torch.utils.tensorboard import SummaryWriter

from torch.cuda.amp import autocast

from daclip import open_clip
from MOE_CLIPbias import MCDB

from torchvision.transforms import Compose, Resize, CenterCrop

##########################################################################
## MB Block
class MB(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type):
        super(MB, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.ssm = MambaLayer(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        prompt = x[1]
        x = x[0]

        residual = x
        x = self.norm1(x)
        x = residual + self.ssm(x, prompt)

        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return {0:x,1:prompt}

##########################################################################
## Mamba
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, prompt_dim=512):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.conv_path = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU()
        )

        self.gate_net = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

    @autocast(enabled=False)
    def forward(self, x, prompt):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, H, W = x.shape

        reversed_x1 = x.clone()

        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        prompt_feat = self.prompt_proj(prompt)  # (B, 512) → (B, C)
        prompt_feat = prompt_feat.view(B, C, 1, 1).expand(-1, -1, H, W)  # (B, C, H, W)

        gate_input = torch.cat([x, prompt_feat], dim=1)  # (B, 2C, H, W)
        gate = self.gate_net(gate_input)  # (B, 1, H, W)

        x1_flat = reversed_x1.reshape(B, C, n_tokens).transpose(-1, -2)
        x1_norm = self.norm(x1_flat)
        x1_mamba = self.mamba(x1_norm)
        x1_mamba = self.norm2(x1_mamba)
        out1 = x1_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        out2 = self.conv_path(reversed_x1)
        out = gate * out1 + (1-gate) * out2

        return out

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Channel-Wise Cross Attention (CA)
class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)


        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

##########################################################################
## H-L Unit
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max = torch.max(x,1,keepdim=True)[0]
        mean = torch.mean(x,1,keepdim=True)
        scale = torch.cat((max, mean), dim=1)
        scale =self.spatial(scale)
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## L-H Unit
class ChannelGate(nn.Module):
    def __init__(self, dim):
        super(ChannelGate, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.max = nn.AdaptiveMaxPool2d((1,1))

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim//16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim//16, dim, 1, bias=False)
        )

    def forward(self, x):
        avg = self.mlp(self.avg(x))
        max = self.mlp(self.max(x))

        scale = avg + max
        scale = F.sigmoid(scale)
        return scale

##########################################################################
## Frequency Modulation Module (FMoM)
class FreRefine(nn.Module):
    def __init__(self, dim):
        super(FreRefine, self).__init__()

        self.SpatialGate = SpatialGate()
        self.ChannelGate = ChannelGate(dim)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, low, high):
        spatial_weight = self.SpatialGate(high)
        channel_weight = self.ChannelGate(low)
        high = high * channel_weight
        low = low * spatial_weight

        out = low + high
        out = self.proj(out)
        return out
    
##########################################################################
## Adaptive Frequency Learning Block (AFLB)
class FreModule(nn.Module):
    def __init__(self, dim, num_heads, bias, in_dim=3):
        super(FreModule, self).__init__()

        self.conv = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.score_gen = nn.Conv2d(2, 2, 7, padding=3)

        self.para1 = nn.Parameter(torch.zeros(dim, 1, 1))
        self.para2 = nn.Parameter(torch.ones(dim, 1, 1))

        self.channel_cross_l = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_h = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)
        self.channel_cross_agg = Chanel_Cross_Attention(dim, num_head=num_heads, bias=bias)

        self.frequency_refine = FreRefine(dim)

        self.rate_conv = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim//8, 2, 1, bias=False),
        )

    def forward(self, x, y):
        _, _, H, W = y.size()
        x = F.interpolate(x, (H,W), mode='bilinear')

        high_feature, low_feature = self.fft(x)
        high_feature = self.channel_cross_l(high_feature, y)
        low_feature = self.channel_cross_h(low_feature, y)

        agg = self.frequency_refine(low_feature, high_feature)
        out = self.channel_cross_agg(y, agg)

        return out * self.para1 + y * self.para2

    def shift(self, x):
        '''shift FFT feature map to center'''
        b, c, h, w = x.shape
        return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2,3))

    def unshift(self, x):
        """converse to shift operation"""
        b, c, h ,w = x.shape
        return torch.roll(x, shifts=(-int(h/2), -int(w/2)), dims=(2,3))

    def fft(self, x, n=128):
        """obtain high/low-frequency features from input"""
        x = self.conv1(x)
        mask = torch.zeros(x.shape).to(x.device)
        h, w = x.shape[-2:]
        threshold = F.adaptive_avg_pool2d(x, 1)
        threshold = self.rate_conv(threshold).sigmoid()

        for i in range(mask.shape[0]):
            h_ = (h//n * threshold[i,0,:,:]).int()
            w_ = (w//n * threshold[i,1,:,:]).int()

            mask[i, :, h//2-h_:h//2+h_, w//2-w_:w//2+w_] = 1

        fft = torch.fft.fft2(x, norm='forward', dim=(-2,-1))
        fft = self.shift(fft)
        
        fft_high = fft * (1 - mask)

        high = self.unshift(fft_high)
        high = torch.fft.ifft2(high, norm='forward', dim=(-2,-1))
        high = torch.abs(high)

        fft_low = fft * mask

        low = self.unshift(fft_low)
        low = torch.fft.ifft2(low, norm='forward', dim=(-2,-1))
        low = torch.abs(low)

        return high, low

##########################################################################
##Taskprompt
class Taskprompt(nn.Module):
    def __init__(self, in_dim, atom_num=32, atom_dim=256):
        super(Taskprompt, self).__init__()
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1))
        self.lastOut = nn.Linear(32, atom_num)
        self.act = nn.GELU()
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        logits = F.softmax(out, -1)
        out = logits @ self.dictionary
        out = self.act(out)

        return out

##########################################################################
##------------------------------ M2Restore -------------------------------

class M2Restore(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias', 
        decoder = True,
    ):

        super(M2Restore, self).__init__()
        atom_dim = 256
        atom_num = 32

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)        
        self.decoder = decoder
        
        if self.decoder:
            self.fre1 = FreModule(dim*2**3, num_heads=heads[2], bias=bias)
            self.fre2 = FreModule(dim*2**2, num_heads=heads[2], bias=bias)
            self.fre3 = FreModule(dim*2**1, num_heads=heads[2], bias=bias)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2

        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[MB(dim=int(dim*2**3), ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.task_prompt = Taskprompt(in_dim=3, atom_num=atom_num, atom_dim=atom_dim)
        self.mcdb1 = MCDB(atom_dim=atom_dim, dim=dim, ffn_expansion_factor=ffn_expansion_factor)
        self.mcdb2 = MCDB(atom_dim=atom_dim, dim=int(dim * 2 ** 1),ffn_expansion_factor=ffn_expansion_factor)
        self.mcdb3 = MCDB(atom_dim=atom_dim, dim=int(dim * 2 ** 2),ffn_expansion_factor=ffn_expansion_factor)

        # da-clip setting
        checkpoint = 'M2Restore/net/daclip/pretrained/daclip_ViT-B-32.pt'
        self.daclip, _ = open_clip.create_model_from_pretrained('da-clip_ViT-B-32', pretrained=checkpoint)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        degradations = ['hazy', 'noisy', 'raindrop', 'rainy','shadowed', 'snowy']
        self.text = tokenizer(degradations).to('cuda')

        self.preprocess = Compose([
            Resize(224, antialias=True),
            CenterCrop(224)
        ])

    def forward(self, inp_img,noise_emb = None):
        with torch.no_grad(), torch.cuda.amp.autocast():
            crop_img = self.preprocess(inp_img)
            text_features = self.daclip.encode_text(self.text)
            image_features, degra_features = self.daclip.encode_image(crop_img, control=True)
            degra_features /= degra_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * degra_features @ text_features.T).softmax(dim=-1)
        de_cls = text_probs

        task_prompt = self.task_prompt(inp_img)
        inp_enc_level1 = self.patch_embed(inp_img)

        task_harmonization_output1, loss_tmp = self.mcdb1(inp_enc_level1, task_prompt, de_cls)
        out_enc_level1 = self.encoder_level1(task_harmonization_output1)
        loss_importance = loss_tmp
        
        inp_enc_level2 = self.down1_2(out_enc_level1)

        task_harmonization_output2, loss_tmp = self.mcdb2(inp_enc_level2, task_prompt, de_cls)
        out_enc_level2 = self.encoder_level2(task_harmonization_output2)
        loss_importance = loss_importance + loss_tmp

        inp_enc_level3 = self.down2_3(out_enc_level2)

        task_harmonization_output3, loss_tmp = self.mcdb3(inp_enc_level3, task_prompt, de_cls)
        out_enc_level3 = self.encoder_level3(task_harmonization_output3)
        loss_importance = loss_importance + loss_tmp

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent({0:inp_enc_level4, 1:image_features.float()})
        latent = latent[0]
        if self.decoder:
            latent = self.fre1(inp_img, latent)
      
        inp_dec_level3 = self.up4_3(latent)

        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        if self.decoder:
            out_dec_level3 = self.fre2(inp_img, out_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            out_dec_level2 = self.fre3(inp_img, out_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        if self.training:
            return out_dec_level1, loss_importance
        else:
            return out_dec_level1