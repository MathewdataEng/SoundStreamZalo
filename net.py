import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from vector_quantize_pytorch import ResidualVQ
from itertools import cycle
from functools import partial
from einops import rearrange, reduce
from torchsummary import summary

def exists(val):
    return val is not None

def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# Encoder Causal Convolution
class CausalConv1d(Module):
    def __init__(self, chan_in, chan_out, kernel_size, pad_mode = 'reflect', **kwargs):
        super().__init__()
        kernel_size = kernel_size
        dilation = kwargs.get('dilation', 1)
        stride = kwargs.get('stride', 1)
        self.pad_mode = pad_mode
        self.causal_padding = dilation * (kernel_size - 1) + (1 - stride)

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, **kwargs)

    def forward(self, x):
#         print(f'This is the input shape {x.shape}')
        x = F.pad(x, (self.causal_padding, 0), mode = self.pad_mode)
        return self.conv(x)

# Decoder Causal Convolution Transpose
class CausalConvTranspose1d(Module):
    def __init__(self, chan_in, chan_out, kernel_size, stride, **kwargs):
        super().__init__()
        self.upsample_factor = stride
        self.padding = kernel_size - 1
        self.conv = nn.ConvTranspose1d(chan_in, chan_out, kernel_size, stride, **kwargs)

    def forward(self, x):
        n = x.shape[-1]

        out = self.conv(x)
        out = out[..., :(n * self.upsample_factor)]

        return out

def ResidualUnit(chan_in, chan_out, dilation, kernel_size = 7, pad_mode = 'reflect'):
    return Residual(Sequential(
        CausalConv1d(chan_in, chan_out, kernel_size, dilation = dilation, pad_mode = pad_mode),
        nn.ELU(),
        CausalConv1d(chan_out, chan_out, 1, pad_mode = pad_mode),
        nn.ELU()
    ))

def EncoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9), pad_mode = 'reflect'):
    it = cycle(cycle_dilations)
    residual_unit = partial(ResidualUnit, pad_mode = pad_mode)

    return nn.Sequential(
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        residual_unit(chan_in, chan_in, next(it)),
        CausalConv1d(chan_in, chan_out, 2 * stride, stride = stride)
    )

def DecoderBlock(chan_in, chan_out, stride, cycle_dilations = (1, 3, 9), pad_mode = 'reflect'):
    even_stride = (stride % 2 == 0)
    padding = (stride + (0 if even_stride else 1)) // 2
    output_padding = 0 if even_stride else 1

    residual_unit = partial(ResidualUnit, pad_mode = pad_mode)

    it = cycle(cycle_dilations)
    return nn.Sequential(
        CausalConvTranspose1d(chan_in, chan_out, 2 * stride, stride = stride),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
        residual_unit(chan_out, chan_out, next(it)),
    )

class FiLM(Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Linear(dim_cond, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
        return x * gamma + beta

class SoundStream(Module):
    def __init__(
        self,
        *,
        channels = 32,
        strides = (2, 4, 5, 8),
        channel_mults = (2, 4, 8, 16),
        codebook_dim = 512,
        codebook_size = 1024,
        rq_num_quantizers = 8,
        input_channels = 1,
        discr_multi_scales = (1, 0.5, 0.25),
        stft_normalized = False,
        enc_cycle_dilations = (1, 3, 9),
        dec_cycle_dilations = (1, 3, 9),
        multi_spectral_window_powers_of_two = tuple(range(6, 12)),
        multi_spectral_n_ffts = 512,
        multi_spectral_n_mels = 64,
        recon_loss_weight = 1.,
        multi_spectral_recon_loss_weight = 1e-5,
        adversarial_loss_weight = 1.,
        target_sample_hz = 16000,
        complex_stft_discr_logits_abs = True,
        pad_mode = 'reflect',
        complex_stft_discr_kwargs: dict = dict()
    ):
        super().__init__()
        self.target_sample_hz = target_sample_hz # for resampling on the fly
        self.single_channel = input_channels == 1
        self.strides = strides
        
        layer_channels = tuple(map(lambda t: t * channels, channel_mults))
        layer_channels = (channels, *layer_channels)
        chan_in_out_pairs = tuple(zip(layer_channels[:-1], layer_channels[1:]))

        encoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(chan_in_out_pairs, strides):
            encoder_blocks.append(EncoderBlock(chan_in, chan_out, layer_stride, enc_cycle_dilations,  pad_mode))
        self.encoder = nn.Sequential(
            CausalConv1d(input_channels, channels, 7, pad_mode = pad_mode),
            *encoder_blocks,
            CausalConv1d(layer_channels[-1], codebook_dim, 3, pad_mode = pad_mode)
        )
        
        self.encoder_film = FiLM(codebook_dim, dim_cond = 2)
        
        self.num_quantizers = rq_num_quantizers
        
        self.codebook_dim = codebook_dim
        
        self.rq = ResidualVQ(
                dim = codebook_dim,
                num_quantizers = rq_num_quantizers,
                codebook_size = codebook_size,
                kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
            )
        self.codebook_size = codebook_size
        self.decoder_film = FiLM(codebook_dim, dim_cond = 2)
        decoder_blocks = []

        for ((chan_in, chan_out), layer_stride) in zip(reversed(chan_in_out_pairs), reversed(strides)):
            decoder_blocks.append(DecoderBlock(chan_out, chan_in, layer_stride, dec_cycle_dilations, pad_mode))


        self.decoder = nn.Sequential(
            CausalConv1d(codebook_dim, layer_channels[-1], 7, pad_mode = pad_mode),
            *decoder_blocks,
            CausalConv1d(channels, input_channels, 7, pad_mode = pad_mode)
        )
    def decode_from_codebook_indices(self, quantized_indices):
        assert quantized_indices.dtype in (torch.long, torch.int32)

        # if quantized_indices.ndim == 3:
            # quantized_indices = rearrange(quantized_indices, 'b n (g q) -> g b n q', g = self.rq_groups)

        x = self.rq.get_output_from_indices(quantized_indices)

        return self.decode(x)
    def decode(self, x, quantize = False):
        if quantize:
            x, *_ = self.rq(x)
        # x = rearrange(x, 'b n c -> b c n')
        return self.decoder(x)
    def tokenize(self, audio):
        self.eval()
        return self.forward(audio, return_codes_only = True)
    def seq_len_multiple_of(self):
        return functools.reduce(lambda x, y: x * y, self.strides)
    def forward(
        self,
        x,
        target = None,
        is_denoising = None, # if you want to learn film conditioners that teach the soundstream to denoise - target would need to be passed in above
        return_encoded = False,
        return_codes_only = False,
        return_discr_loss = False,
        return_discr_losses_separately = False,
        return_recons_only = False,
        input_sample_hz = None
    ):
        
        orig_x = x.clone()
        # print(f'This is the original x = {orig_x}')
        
        x = self.encoder(x)
        
        x = rearrange(x, 'b c n -> b n c')
        # print(f'This is the encoded x = {x}')
        # print(f'This is the encoded x shape = {x.shape}')
        if exists(is_denoising):
            denoise_input = torch.tensor([is_denoising, not is_denoising], dtype = x.dtype) # [1, 0] for denoise, [0, 1] for not denoising
            x = self.encoder_film(x, denoise_input)
        x, indices,commit_loss = self.rq(x)
        if return_codes_only:
            return indices
        if return_encoded:
            # indices = rearrange(indices, 'g b n q -> b n (g q)')
            indices = int(indices)
            return x, indices, commit_loss
        # print(f'After residual vector quantization x = {x}')
        # print(f'After residual vector quantization x shape = {x.shape}')
        ### Decoder part
        if exists(is_denoising):
            x = self.decoder_film(x, denoise_input)
            
        x = rearrange(x, 'b n c -> b c n')
        
        recon_x = self.decoder(x)
        
#         if return_recons_only:
#         recon_x, = unpack(recon_x, ps, '* c n')
        # print(f'Recontruction x = {recon_x}')
        return recon_x


# Wave-based Discriminator


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class WaveDiscriminatorBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                WNConv1d(in_channels=1, out_channels=16, kernel_size=15),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=16, out_channels=64, kernel_size=41,
                         stride=4, padding=20, groups=4),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=64, out_channels=256, kernel_size=41,
                         stride=4, padding=20, groups=16),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=256, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=64),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=41,
                         stride=4, padding=20, groups=256),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            nn.Sequential(
                WNConv1d(in_channels=1024, out_channels=1024, kernel_size=5,
                         stride=1, padding=2),
                nn.LeakyReLU(negative_slope=0.2)
            ),
            WNConv1d(in_channels=1024, out_channels=1, kernel_size=3, stride=1,
                     padding=1)
        ])
    
    def features_lengths(self, lengths):
        return [
            lengths,
            torch.div(lengths+3, 4, rounding_mode="floor"),
            torch.div(lengths+15, 16, rounding_mode="floor"),
            torch.div(lengths+63, 64, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor"),
            torch.div(lengths+255, 256, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map


class WaveDiscriminator(nn.Module):
    def __init__(self, num_D, downsampling_factor):
        super().__init__()
        
        self.num_D = num_D
        self.downsampling_factor = downsampling_factor
        
        self.model = nn.ModuleDict({
            f"disc_{downsampling_factor**i}": WaveDiscriminatorBlock()
            for i in range(num_D)
        })
        self.downsampler = nn.AvgPool1d(kernel_size=4, stride=2, padding=1,
                                        count_include_pad=False)
    
    def features_lengths(self, lengths):
        return {
            f"disc_{self.downsampling_factor**i}": self.model[f"disc_{self.downsampling_factor**i}"].features_lengths(torch.div(lengths, 2**i, rounding_mode="floor")) for i in range(self.num_D)
        }
    
    def forward(self, x):
        results = {}
        for i in range(self.num_D):
            disc = self.model[f"disc_{self.downsampling_factor**i}"]
            results[f"disc_{self.downsampling_factor**i}"] = disc(x)
            x = self.downsampler(x)
        return results


# STFT-based Discriminator

class ResidualUnit2d(nn.Module):
    def __init__(self, in_channels, N, m, s_t, s_f):
        super().__init__()
        
        self.s_t = s_t
        self.s_f = s_f

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=N,
                kernel_size=(3, 3),
                padding="same"
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=N,
                out_channels=m*N,
                kernel_size=(s_f+2, s_t+2),
                stride=(s_f, s_t)
            )
        )
        
        self.skip_connection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=m*N,
            kernel_size=(1, 1), stride=(s_f, s_t)
        )

    def forward(self, x):
        return self.layers(F.pad(x, [self.s_t+1, 0, self.s_f+1, 0])) + self.skip_connection(x)


class STFTDiscriminator(nn.Module):
    def __init__(self, C, F_bins):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(7, 7)),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=32,  N=C,   m=2, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=2*C, N=2*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, N=4*C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=4*C, N=4*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C, N=8*C, m=1, s_t=1, s_f=2),
                nn.ELU()
            ),
            nn.Sequential(
                ResidualUnit2d(in_channels=8*C,  N=8*C, m=2, s_t=2, s_f=2),
                nn.ELU()
            ),
            nn.Conv2d(in_channels=16*C, out_channels=1,
                      kernel_size=(F_bins//2**6, 1))
        ])
    
    def features_lengths(self, lengths):
        return [
            lengths-6,
            lengths-6,
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-5, 2, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths-3, 4, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor"),
            torch.div(lengths+1, 8, rounding_mode="floor")
        ]

    def forward(self, x):
        feature_map = []
        for layer in self.layers:
            x = layer(x)
            feature_map.append(x)
        return feature_map


    
# if __name__ == "__main__":

#     model = SoundStream()
#     PATH = './Checkpoints/200.pt'
#     model.load_state_dict(torch.load(PATH))
#     x = torch.randn(1,1,16000)
#     G_x = model(x)
#     print(f'This is Input from Soundstream  {x}')
#     print(f'This is Output from Soundstream  {G_x}')
#     print(f'This is Distance from Soundstream output vs input =  {torch.cdist(x, G_x)}')
#     W, H = 1024, 256
#     wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
#     stft_disc = STFTDiscriminator(C=1, F_bins=W//2)
#     features_wave_disc_x = wave_disc(G_x)
#     # print(f'This is output from wave discriminator {features_wave_disc_x}')
#     s_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024),return_complex=True)
#     # print(f'This is output from stft discriminator {s_x}')
#     s_x = torch.view_as_real(s_x).permute(2, 0 , 1)
#     features_stft_disc_x = stft_disc(s_x)
#     # print(f'This is output from stft discriminator {features_stft_disc_x}')
import torchaudio
if __name__ == "__main__":

    model = SoundStream()
    # PATH = './Checkpoints/200.pt'
    # model.load_state_dict(torch.load(PATH))
    # Load the model
    state_dict = torch.load('./Checkpoints/200.pt')

    # Remove "module." prefix
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load the modified state dictionary
    model.load_state_dict(new_state_dict)

    x,sr = torchaudio.load('./test_audio.mp3')
    target_length = 160000
    if x.size(1) > target_length:
        x = x[:, :target_length]

    x = torch.unsqueeze(x, dim=1)

    G_x = model(x)
    print(f'This is Input from Soundstream  {x}')
    print(f'This is Output from Soundstream  {G_x}')
    
    print(f'This is Distance from Soundstream output vs input =  {torch.cdist(x, G_x)}')
    final = torch.squeeze(G_x, dim=1)
    path = './reconstruction_audio.mp3'
    torchaudio.save(path, final, sr)
    W, H = 1024, 256
    wave_disc = WaveDiscriminator(num_D=3, downsampling_factor=2)
    stft_disc = STFTDiscriminator(C=1, F_bins=W//2)
    features_wave_disc_x = wave_disc(G_x)
    # print(f'This is output from wave discriminator {features_wave_disc_x}')
    s_x = torch.stft(G_x.squeeze(), n_fft=1024, hop_length=256, window=torch.hann_window(window_length=1024),return_complex=True)
    # print(f'This is output from stft discriminator {s_x}')
    s_x = torch.view_as_real(s_x).permute(2, 0 , 1)
    features_stft_disc_x = stft_disc(s_x)
    # print(f'This is output from stft discriminator {features_stft_disc_x}')
