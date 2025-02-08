from typing import Tuple, Union
import torch
from torch import nn
import numpy as np
from .prompt import VideoSpecificPrompt
from .vvt import VideoVisionTransformer
import sys
import warnings

sys.path.append("../")
from clip.model import CLIP, LayerNorm, Transformer
import clip


class CLAVER(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # video
                 # temporal_layers: int,
                 T=8,
                 droppath=0.,
                 mask_mode='KMT', # ['KMT', 'KMCT']
                 # other
                 use_cache=True,
                 use_checkpoint=False,
                 ):
        super().__init__(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )
        self.use_cache = use_cache
        self.T = T
        self.L = (image_resolution // vision_patch_size) ** 2 + 1
        dpr = [x.item() for x in torch.linspace(0, droppath, vision_layers)] if droppath > 0. else None

        vision_heads = vision_width // 64
        self.visual = VideoVisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            droppath=dpr,
            T=T,
            use_checkpoint=use_checkpoint,
            attn_mask_t=self.build_temporal_mask() if mask_mode == 'KMT' else self.build_causal_temporal_mask()
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.cache_text_features = None

        self.initialize_parameters()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'positional_embedding'}

    def build_temporal_mask(self, fill_value='-inf'):
        '''
        attn: [B, H, TL, TL]
        L: patch nums or node nums
        T: time slots
        '''
        T = self.T
        L = self.L
        mask = torch.ones((L, L)) - torch.eye(L)
        diag = torch.eye(T)
        attn_mask = self.kronecker(diag, mask)
        if fill_value == '-inf':
            attn_mask[attn_mask == 1] = - torch.inf
        elif fill_value == 'zero':
            attn_mask[attn_mask == 1] = 0.
        return attn_mask
    
    def build_causal_temporal_mask(self, fill_value='-inf'):
        '''
        attn: [B, H, TL, TL]
        L: patch nums
        T: time slots
        '''
        T = self.T
        L = self.L
        mask = torch.ones((L, L)) - torch.eye(L)
        diag = torch.eye(T)
        attn_mask = self.kronecker(diag, mask)

        causal_mask = torch.ones((L, L))
        up_triu = self.upper_triangle_mask(T)
        attn_causal_mask = self.kronecker(up_triu, causal_mask)
        attn_mask += attn_causal_mask
        if fill_value == '-inf':
            attn_mask[attn_mask == 1] = - torch.inf
        elif fill_value == 'zero':
            attn_mask[attn_mask == 1] = 0.
        return attn_mask
    
    def upper_triangle_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask
    
    def kronecker(self, A, B):
        AB = torch.einsum("ab,cd->acbd", A, B)
        AB = AB.view(A.size(0) * B.size(0), A.size(1) * B.size(1))
        return AB

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)
        eos_indx = text.argmax(dim=-1)
        K, N1, C = x.shape  # [num_classes, content_length, embed_dim]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(K, -1)  # [num_classes, embed_dim]
        return x

    def encode_video(self, image):
        b, t, c, h, w = image.size()
        image = image.reshape(-1, c, h, w)  # [bt,c,h,w]

        cls_features, image_features = self.encode_image(image) # [bt,d] [bt,l,d]

        cls_features = cls_features.view(b, t, -1)
        image_features = image_features.view(b, t, -1, image_features.size(-1))

        return cls_features, image_features # [b,t,d] [b,t,l,d]

    def cache_text(self, text):
        self.eval()
        with torch.no_grad():
            if self.cache_text_features is None:
                self.cache_text_features = self.encode_text(text)
        self.train()
        return self.cache_text_features

    def forward(self, image, text):
        b = image.shape[0]
        A, C, _ = text.shape
        text = text.reshape(A * C, -1)

        video_features, _ = self.encode_video(image)  # [b,t,d] [b,t,l,d]

        if self.use_cache:
            text_features = self.cache_text(text)  # [A*num_classes, embed_dim]
        else:
            text_features = self.encode_text(text)  # [A*num_classes, embed_dim]

        text_features = text_features.unsqueeze(0).expand(b, -1, -1)

        video_features = video_features / video_features.norm(dim=-1,keepdim=True) # [b,t,d]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True) # [b,A*num_classes,embed_dim]
        logit_scale = self.logit_scale.exp()

        video_features = video_features.mean(1, keepdim=False)
        logits = torch.einsum("bd,bkd->bk", video_features, logit_scale * text_features)

        logits = logits.reshape(b, A, -1) # [b,A,num_classes]
        return logits


def build_model(state_dict: dict, T=8, droppath=0., use_checkpoint=False, logger=None, use_cache=True, mask_mode='KMT'):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)

        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLAVER(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        T=T, droppath=droppath, use_checkpoint=use_checkpoint, use_cache=use_cache, mask_mode=mask_mode
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"load pretrained CLIP: {msg}")

    return model.eval()


def load(model_path, name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
         jit=True, T=8, droppath=0., use_checkpoint=False, logger=None, use_cache=True, mask_mode='KMT'
         ):
    if model_path is None:
        model_path = clip._download(clip._MODELS[name])
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), T=T, droppath=droppath,
                        use_checkpoint=use_checkpoint, logger=logger,
                        use_cache=use_cache, mask_mode=mask_mode
                        )
    if str(device) == "cpu":
        model.float()
    return model, model.state_dict()

