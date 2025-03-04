import json
import math
import os
import re
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image, ImageOps
from transformers import AutoTokenizer

class Tic:
    def __init__(self):
        self.last_time = time.perf_counter()

    def __call__(self):
        current_time = time.perf_counter()
        elapsed_time = current_time - self.last_time
        self.last_time = current_time
        return elapsed_time

class TrainingCallback:
    def __init__(self, lora_cfg, lr_schedule, batch_indices, sum_every=3):
        self.batch_indices = batch_indices
        self.lora_cfg = lora_cfg
        self.adapter_path = lora_cfg['adapter_path']
        self.lr_schedule = lr_schedule
        self.sum_every = min(sum_every, len(batch_indices))
        self.current_step = 0
        self.sum_loss = .0
        self.best_loss = math.inf
        self.train_log = {'step_i': [], 'step_loss': [], 'avg_i': [], 'avg_loss': []}
        self.start_time = time.perf_counter()
        os.makedirs(self.adapter_path, exist_ok=True)

    def __call__(self, model, lvalue):
        self.current_step += 1
        step_loss = lvalue.item()
        print(f'- Step loss at step {self.current_step}: {step_loss:.2f}')
        self.train_log['step_i'].append(self.current_step)
        self.train_log['step_loss'].append(step_loss)
        self.sum_loss += step_loss

        if self.current_step % self.sum_every == 0:
            avg_loss = self.sum_loss / self.sum_every
            self.sum_loss = 0.0
            self.train_log['avg_i'].append(self.current_step)
            self.train_log['avg_loss'].append(avg_loss)
            print(f'Avg loss at step {self.current_step}: {avg_loss:.2f}')
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                mx.save_safetensors(f'{self.adapter_path}/adapters.safetensors', dict(tree_flatten(model.trainable_parameters())))

    def end_log(self):
        train_log = self.train_log
        train_log['train_time'] = time.perf_counter() - self.start_time
        with open(f'{self.adapter_path}/adapter_config.json', "w") as f:
            json.dump(self.lora_cfg, f, indent=4)
        with open(f'{self.adapter_path}/adapter_train_log.json', "w") as f:
            json.dump(train_log, f, indent=4)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(train_log['step_i'], train_log['step_loss'], color='b', alpha=0.5, label='Step Loss')
        ax1.plot(train_log['avg_i'], train_log['avg_loss'], color='r', label='Avg Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax2.plot(self.lr_schedule)
        ax2.ticklabel_format(axis='y', style='sci')
        ax2.set_title('Learning Rate Schedule')
        batch_indices = np.array(self.batch_indices)
        batch_numbers = np.arange(len(batch_indices))
        x = np.repeat(batch_numbers, [len(sublist) for sublist in batch_indices])
        y = np.concatenate(batch_indices)
        ax3.scatter(x,y, color='b', marker='.', alpha=0.5)
        ax3.set_title('Batch Indices')
        plt.tight_layout()
        fig.savefig(f'train_log_{self.current_step}_steps_in_{train_log["train_time"]:.0f}_sec.png')
        print(f"Training log saved to {self.adapter_path}")
        print(f"Total training time: {train_log['train_time']:.2f} seconds")

class LoRALinear(nn.Module): # https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/lora.py
    @staticmethod
    def from_linear(
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            alpha=alpha,
            dropout=dropout,
            scale=scale,
        )
        lora_lin.linear = linear
        return lora_lin

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        scale: float = 10.0,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale * (alpha / r)
        scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        z = y + (self.scale * z)
        return z.astype(x.dtype)

class ClipAttention(nn.Module):
    def __init__(self, dims, num_heads, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dims // num_heads) ** -0.5
        self.q_proj = nn.Linear(dims, dims, bias=bias)
        self.k_proj = nn.Linear(dims, dims, bias=bias)
        self.v_proj = nn.Linear(dims, dims, bias=bias)
        self.out_proj = nn.Linear(dims, dims, bias=bias)

    def __call__(self, x):
        B, L = x.shape[:2]
        queries, keys, values = (proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3) for proj in (self.q_proj, self.k_proj, self.v_proj))
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        return self.out_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))

class ClipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = nn.gelu_fast_approx
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, x):
        return self.fc2(self.activation_fn(self.fc1(x)))

class ClipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = ClipAttention(config.hidden_size, config.num_attention_heads, bias=True)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ClipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))

class ClipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [ClipEncoderLayer(config) for _ in range(config.num_hidden_layers)]

class ClipEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = mx.zeros(config.hidden_size)
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def __call__(self, x):
        batch_size = x.shape[0]
        patch_embeddings = self.patch_embedding(x)
        patch_embeddings = mx.flatten(patch_embeddings, start_axis=1, end_axis=2)
        embed_dim = patch_embeddings.shape[-1]
        cls_embeddings = mx.broadcast_to(self.class_embedding, (batch_size, 1, embed_dim))
        position_ids = mx.arange(self.num_positions)[None]
        embeddings = mx.concatenate((cls_embeddings, patch_embeddings), axis=1)
        embeddings += self.position_embedding(position_ids)
        return embeddings

class ClipModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = ClipEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size)
        self.encoder = ClipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x):
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)
        for l in self.encoder.layers[:-1]:
            x = l(x)
        return x[:, 1:]

class ClipVModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = ClipModel(config)

class Phi3FProcessor:
    def __init__(self, local_dir, return_mx=True):
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.return_mx = return_mx

    def _tokenize(self, texts):
        if isinstance(texts, str):
            return {'input_ids': mx.array(self.tokenizer(texts).input_ids)[None]}
        input_ids = self.tokenizer(texts).input_ids
        max_length = max(len(sublist) for sublist in input_ids)
        position_ids =[[1]*(max_length-len(sublist)) + list(range(len(sublist))) for sublist in input_ids]
        attention_masks = [[0]*(max_length-len(sublist)) + [1]*len(sublist) for sublist in input_ids]
        input_ids = [[0]*(max_length-len(sublist)) + sublist for sublist in input_ids]
        if self.return_mx:
            input_ids = mx.array(input_ids)
            position_ids = mx.array(position_ids)
            attention_masks = mx.array(attention_masks)
        return {'input_ids':input_ids, 'pids':position_ids, 'mask':attention_masks}

    def __call__(self, texts, images=None):
        if images is not None:
            print(f'WARNING: You are using phi3_mini_128k. Use phi3_v for VLM tasks.')
        return self._tokenize(texts)

class Phi3VProcessor(Phi3FProcessor):
    def __init__(self, local_dir, return_mx=True):
        super().__init__(local_dir, return_mx)
        self.img_processor = Phi3VImageProcessor()

    def __call__(self, texts, images=None):
        if images is None:
            return self._tokenize(texts)
        image_inputs = self.img_processor(images)
        return self._merge(image_inputs, texts)

    def _merge(self, images, texts):
        pattern = r"<\|image_\d+\|>"
        prompt_chunks = self.tokenizer(re.split(pattern, texts)).input_ids
        num_img_tokens = images['num_img_tokens']
        images, image_sizes = images['pixel_values'], images['image_sizes']
        image_tags = re.findall(pattern, texts)
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
        image_ids_pad = [[-iid]*num_img_tokens[iid-1] for iid in image_ids]
        image_ids_pad = image_ids_pad + [[]] if len(prompt_chunks) > len(image_ids_pad) else image_ids_pad
        input_ids = []
        for chunk, pad in zip(prompt_chunks, image_ids_pad):
            input_ids.extend(chunk)
            input_ids.extend(pad)
        input_ids = np.array(input_ids)[None]
        positions = np.argwhere(input_ids < 0)
        return {"input_ids": mx.array(input_ids),
                "pixel_values": mx.array(images),
                "image_sizes": mx.array(image_sizes),
                "positions": mx.array(positions)}

class Phi3VImageProcessor:
    def __init__(self):
        self.num_crops=16
        self.image_mean=np.array([0.48145466, 0.4578275, 0.40821073])
        self.image_std=np.array([0.26862954, 0.26130258, 0.27577711])

    def __call__(self, images):
        def HD_transform(img):
            img = img.convert('RGB')
            w, h = img.size
            trans = False
            if w < h:
                img = img.transpose(Image.TRANSPOSE)
                trans = True
                w, h = img.size
            scale = int(np.sqrt(self.num_crops * w / h))
            img = img.resize([int(scale * 336), int(scale * 336 * h / w)], Image.BILINEAR)
            def pad_to_336(b):
                _, h = b.size
                diff_height = int(np.ceil(h / 336) * 336) - h
                top_padding = int(diff_height/2)
                bottom_padding = diff_height - top_padding
                b = ImageOps.expand(b, border=(0, top_padding, 0, bottom_padding), fill=(255, 255, 255))
                return b
            img = pad_to_336(img)
            img = img.transpose(Image.TRANSPOSE) if trans else img
            img = ((np.array(img) / 255.0 - self.image_mean) / self.image_std).transpose(2,0,1)
            return img
        def pad_to_max_num_crops_tensor(images, max_crops=17):
            B, _, H, W = images.shape
            if B < max_crops:
                pad = np.zeros((max_crops - B, 3, H, W))
                images = np.concatenate([images, pad], axis=0)
            return images
        hd_images = [HD_transform(img) for img in images]
        shapes = [[im.shape[1], im.shape[2]] for im in hd_images]
        num_img_tokens = [int((h//336*w//336+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]
        # global_image = [torch.nn.functional.interpolate(torch.from_numpy(im[None]), size=(336, 336), mode='bicubic').numpy() for im in hd_images]
        global_image = [self.interpolate_336(im[None]) for im in hd_images]
        hd_images_reshape = [im
            .reshape(1, 3, h//336, 336, w//336, 336)
            .transpose(0,2,4,1,3,5)
            .reshape(-1, 3, 336, 336)
            for im, (h, w) in zip(hd_images, shapes)]
        hd_images_reshape = [np.concatenate([_global_image, _im], axis=0) for _global_image, _im in zip(global_image, hd_images_reshape)]
        image_transformed = np.stack([pad_to_max_num_crops_tensor(im) for im in hd_images_reshape], axis=0)
        return {"pixel_values": image_transformed, "image_sizes": shapes, "num_img_tokens": num_img_tokens}

    @staticmethod
    def interpolate_336(input):
        def get_weights_and_indices(scale, out_size, in_size):
            def cubic(x):
                abs_x = np.abs(x)
                abs_x2 = abs_x ** 2
                abs_x3 = abs_x ** 3
                f = ((1.5 * abs_x3 - 2.5 * abs_x2 + 1) * (abs_x <= 1) +
                    (-0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2) * ((abs_x > 1) & (abs_x <= 2)))
                return f
            kernel_radius = 2
            kernel_width = kernel_radius * 2
            out_coordinates = np.linspace(0, in_size - 1, out_size)
            in_coordinates = out_coordinates / scale
            left_indices = np.floor(in_coordinates - 0.5).astype(np.int32)
            right_indices = left_indices + 1
            left_indices = np.clip(left_indices, 0, in_size - 1)
            right_indices = np.clip(right_indices, 0, in_size - 1)
            weights = np.zeros((out_size, kernel_width), dtype=np.float32)
            indices = np.zeros((out_size, kernel_width), dtype=np.int32)
            for i in range(out_size):
                indices[i, 0] = left_indices[i]
                indices[i, 1] = right_indices[i]
                weights[i, 0] = cubic(in_coordinates[i] - left_indices[i])
                weights[i, 1] = cubic(right_indices[i] - in_coordinates[i])
                weight_sum = weights[i].sum()
                if weight_sum != 0:
                    weights[i] /= weight_sum
            return weights, indices
        N, C, H, W = input.shape
        out_hw = 336
        output = np.zeros((N, C, out_hw, out_hw), dtype=input.dtype)
        h_weights, h_indices = get_weights_and_indices(out_hw / H, out_hw, H)
        w_weights, w_indices = get_weights_and_indices(out_hw / W, out_hw, W)
        for n in range(N):
            for c in range(C):
                for i in range(out_hw):
                    for j in range(out_hw):
                        h_kernel = input[n, c, h_indices[i]]
                        w_kernel = h_kernel[:, w_indices[j]]
                        output[n, c, i, j] = np.sum(h_weights[i][:, None] * w_weights[j] * w_kernel)
        return output

class Phi3ImageEmbedding(nn.Module):
    CLIP_VIT_LARGE_PATCH14_336_CONFIG = SimpleNamespace(
        hidden_size=1024,
        image_size=336,
        intermediate_size=4096,
        layer_norm_eps=1e-05,
        num_attention_heads=16,
        num_channels=3,
        num_hidden_layers=24,
        patch_size=14,
        )
    def __init__(self, config):
        super().__init__()
        self.img_processor = ClipVModel(self.CLIP_VIT_LARGE_PATCH14_336_CONFIG)
        self.image_dim_out = image_dim_out = config.img_processor['image_dim_out']
        self.glb_GN = mx.zeros([1, 1, image_dim_out * 4])
        self.sub_GN = mx.zeros([1, 1, 1, image_dim_out * 4])
        self.img_projection = [nn.Linear(image_dim_out * 4, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)]

    def __call__(self, txt_embeds, img_embeds, img_sizes, positions):
        B = img_embeds.shape[0]
        img_sizes, positions = (img_sizes // 336).tolist(), positions.tolist()
        img_features = self.img_processor.vision_model(img_embeds.reshape(-1, *img_embeds.shape[2:]).transpose(0, 2, 3, 1))
        img_features = img_features.reshape(B, -1, *img_features.shape[1:])
        C, H = self.image_dim_out, int(img_features.shape[2] ** 0.5)
        output_imgs, output_len = [], []
        for _bs in range(B):
            h, w = img_sizes[_bs]
            B_ = h * w
            def _reshape_and_concatenate(img, shape, tile_shape):
                return mx.concatenate([img.reshape(shape).transpose(0, 1, 3, 2, 4, 5).reshape(tile_shape), mx.tile(self.sub_GN, (1, tile_shape[1], 1, 1))], axis=2).reshape(1, -1, 4 * C)
            glb_img = _reshape_and_concatenate( img_features[_bs, :1], (1, H//2, 2, H//2, 2, C), (1, H//2, H//2, 4*C) )
            sub_img = _reshape_and_concatenate( img_features[_bs, 1:B_+1], (B_, H//2, 2, H//2, 2, C), (1, h*12, w*12, 4*C) )
            x = mx.concatenate([sub_img, self.glb_GN, glb_img], axis=1)
            for l in self.img_projection:
                x = l(x)
            output_imgs.append(x)
            output_len.append(int((h*w + 1) * 144 + 1 + (h + 1) * 12))
        idx = 0
        for i, cnt in enumerate(output_len):
            # txt_embeds[positions[idx][0], positions[idx][1] : positions[idx][1] + cnt] = output_imgs[i]
            ###
            # target = txt_embeds[positions[idx][0], positions[idx][1] : positions[idx][1] + cnt].shape
            # txt_embeds[positions[idx][0], positions[idx][1] : positions[idx][1] + cnt] = output_imgs[i].reshape(target)
            ###
            txt_embeds[positions[idx][0], positions[idx][1] : positions[idx][1] + cnt] = output_imgs[i][0]
            idx += cnt
        return txt_embeds

@mx.compile
def _rotate_half(x, cos, sin):
    midpoint = x.shape[-1] // 2
    x1, x2 = x[..., :midpoint], x[..., midpoint:]
    result = (x * cos) + (mx.concatenate([-x2, x1], axis = -1) * sin)
    return result

class Phi3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.head_dim = head_dim = config.hidden_size // n_heads
        self.scale = head_dim**-0.5
        self.chop_1 = chop_1 = self.n_heads * self.head_dim
        self.chop_2 = chop_1 + self.n_kv_heads * self.head_dim
        op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
        self.qkv_proj = nn.Linear(dim, op_size, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def __call__(self, x, cache, cos, sin, mask, n_beam):
        B, L, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = mx.split(qkv, [self.chop_1, self.chop_2], axis=-1)
        q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if n_beam > 1:
            sin = mx.repeat(sin, repeats=n_beam, axis=0)
            cos = mx.repeat(cos, repeats=n_beam, axis=0)
            mask = mx.repeat(mask, repeats=n_beam, axis=0)
        q = _rotate_half(q, cos, sin)
        k = _rotate_half(k, cos, sin)
        k, v = cache(k, v, n_beam)
        w = (q * self.scale) @ k.transpose(0, 1, 3, 2)
        w += mask
        w = mx.softmax(w, axis=-1)
        o = w @ v
        # o = mx.fast.scaled_dot_product_attention(q,k,v,scale=self.scale,mask=mask)
        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o).astype(qkv.dtype)

class Phi3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x):
        x = self.gate_up_proj(x)
        gate, x = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * x)

class Phi3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Phi3Attention(config)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, cache, cos, sin, mask, n_beam):
        r = self.self_attn(self.input_layernorm(x), cache, cos, sin, mask, n_beam)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r

class SuRoPE(nn.Module):
    def __init__(self, config, L_all, pids):
        super().__init__()
        dim = config.hidden_size // config.num_attention_heads
        scaling_factor = math.sqrt(1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings))
        su_factor = config.rope_scaling["long_factor"] if L_all > config.original_max_position_embeddings else config.rope_scaling["short_factor"]
        if pids is None:
            position_ids = mx.arange(L_all, dtype=mx.float32)[None]
        else:
            extended_pids = pids[:, -1][:, None] + 1 + mx.arange(L_all-pids.shape[1], dtype=mx.float32)[None, :]
            position_ids = mx.concatenate([pids, extended_pids], axis=1)
        position_ids_expanded = position_ids[:, None, :]
        inv_freq = 1.0 / (mx.array(su_factor, dtype=mx.float32) * config.rope_theta**(mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        inv_freq_expanded = mx.repeat(inv_freq[None, :, None], position_ids.shape[0], axis=0)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self._cos = mx.expand_dims(mx.cos(emb) * scaling_factor, axis=1)
        self._sin = mx.expand_dims(mx.sin(emb) * scaling_factor, axis=1)

    def __call__(self, past_L, new_L):
        return self._cos[:,:,past_L:past_L+new_L,:], self._sin[:,:,past_L:past_L+new_L,:]

class KVCache:
    def __init__(self, config, x, max_tokens):
        self.max_tokens = max_tokens
        self.use_quantized_cache = getattr(config, "use_quantized_cache", False)
        self.offset = 0
        self.shape = (2, x.shape[0], config.num_key_value_heads, x.shape[1]+max_tokens, config.hidden_size // config.num_key_value_heads)
        self.kv = None
        if self.use_quantized_cache or max_tokens < 1:
            self.keys = []
            self.values = []

    def __call__(self, keys, values, n_beam):
        if self.max_tokens < 1:
            return keys, values
        if n_beam > 1:
            if self.use_quantized_cache:
                raise NotImplementedError('Beam Search is not yet compatible with Quantized Cache')
            kv = mx.repeat(self.kv[:,:,:,:self.offset,:], repeats=n_beam, axis=1)
            return mx.concatenate([kv[0], keys], axis=-2), mx.concatenate([kv[1], values], axis=-2)
        if self.use_quantized_cache:
            self.offset += values.shape[2]
            _, B, N, _, D = self.shape
            if self.kv is None:
                self.kv = (mx.quantize(keys.reshape((B*N,-1)), group_size=32), mx.quantize(values.reshape((B*N,-1)), group_size=32))
                return keys, values
            self.keys.append(keys)
            self.values.append(values)
            k_cache = mx.dequantize(*self.kv[0], group_size=32).reshape((B, N, -1, D))
            v_cache = mx.dequantize(*self.kv[1], group_size=32).reshape((B, N, -1, D))
            keys = mx.concatenate([k_cache] + self.keys, axis=2)
            values = mx.concatenate([v_cache] + self.values, axis=2)
            return keys, values
        else:
            if self.kv is None:
                self.kv = mx.zeros(self.shape, dtype=keys.dtype)
            new_offset = self.offset + keys.shape[2]
            self.kv[0,:,:,self.offset:new_offset,:] = keys
            self.kv[1,:,:,self.offset:new_offset,:] = values
            self.offset = new_offset
            return self.kv[0,:,:,:new_offset,:], self.kv[1,:,:,:new_offset,:]

class Mask4D:
    def __init__(self, L_all, mask):
        mask_4d = mx.triu(mx.full((L_all, L_all), -mx.inf), k=1)[None, None]
        if mask is not None:
            pad_len = L_all - mask.shape[-1]
            mask = mx.pad(mask, ((0,0),(0,pad_len)), 1)
            mask = mx.expand_dims(mask, (1,2))
            mask = mask*mask.transpose(0,1,3,2)
            mask = mx.where(mask==1, 0, -mx.inf)
            mask_4d += mask
        self.mask_4d = mask_4d

    def __call__(self, past_L, L):
        return self.mask_4d[:,:,past_L:L+past_L,:L+past_L]

class Phi3F(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.vision_embed_tokens = None
        self.layers = [Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.use_quantized_cache= getattr(config, "use_quantized_cache", False)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers

    def __call__(self, input_ids, pixel_values, image_sizes, positions, cache, pids, mask, max_tokens, advance_offset, n_beam):
        x = self.embed_tokens(input_ids)
        if pixel_values is not None and self.vision_embed_tokens:
            x = self.vision_embed_tokens(x, pixel_values, image_sizes, positions)
        if cache is None:
            cache = [KVCache(self.config, x, max_tokens) for _ in range(self.num_hidden_layers)]
            self._masker = Mask4D(x.shape[1]+max_tokens, mask)
            self._roper = SuRoPE(self.config, x.shape[1]+max_tokens, pids)
        past_L, new_L = cache[0].offset, x.shape[1]
        mask = self._masker(past_L, new_L)
        cos, sin = self._roper(past_L, new_L)
        for i, l in enumerate(self.layers):
            x = l(x, cache[i], cos, sin, mask, n_beam)
        if advance_offset is not None:
            for c in cache:
                c.offset = past_L + advance_offset
        return self.norm(x), cache

class Phi3V(Phi3F):
    def __init__(self, config):
        super().__init__(config)
        self.vision_embed_tokens = Phi3ImageEmbedding(config)

class Phi3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Phi3F(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(self, input_ids, pixel_values=None, image_sizes=None, positions=None, cache=None, pids=None, mask=None, max_tokens=0, advance_offset=None, n_beam=1):
        x, cache = self.model(input_ids, pixel_values, image_sizes, positions, cache, pids, mask, max_tokens, advance_offset, n_beam)
        return self.lm_head(x), cache, x

    @property
    def layers(self):
        return self.model.layers

class Phi3VForCausalLM(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Phi3V(config)




ID_EOS = 32007
ID_ASS = 32001


class Streamer:
    def __init__(self, processor, stream, mute):
        self.tokenizer = processor.tokenizer
        self.mute = mute
        self.stream = stream and (not mute)
        self.list_tokens = []
        self.idx_sofar = 0
    def __call__(self, token):
        if not self.stream:
            self.list_tokens.append(token)
            return None
        if token.shape[0] > 1:
            self.list_tokens.append(token)
            self.stream = False
            return None
        self.list_tokens.append(token.item())
        txt = self.tokenizer.decode(self.list_tokens)
        idx_split = txt.rfind(' ', self.idx_sofar)
        if idx_split > 0:
            print(txt[self.idx_sofar:idx_split], end = '', flush=True)
            self.idx_sofar = idx_split
    def end(self):
        if self.stream:
            txt = self.tokenizer.decode(self.list_tokens)
            print(txt[self.idx_sofar:], '\n', flush=True)
            return txt, len(self.list_tokens)
        else:
            arr_tokens = mx.concatenate(self.list_tokens, axis=1)
            list_txt = self.tokenizer.batch_decode([(i[:i.index(ID_EOS)+1] if ID_EOS in i else i) for i in arr_tokens.tolist()])
            if not self.mute:
                for i, gen in enumerate(list_txt):
                    print(f'\n< Generated text for prompt #{i} >\n{gen}')
            return list_txt, arr_tokens.size

class LogitStopper:
    def __init__(self, max_tokens, early_stop):
        self.step = 0
        self.early_stop = early_stop if isinstance(early_stop, int) and (early_stop < max_tokens) else False
        self.log_prob_sum = 0.0
        self.best_eos_sofar = -mx.inf
        self.log_prob_sum_at_best_eos = 0.0
    def __call__(self, logits):
        if not self.early_stop:
            return False
        if logits.shape[0] > 1:
            self.early_stop = False
            return False
        log_prob = nn.log_softmax(logits[:,-1,:])
        log_prob_best = mx.max(log_prob, axis=-1).item()
        log_prob_eos = log_prob[:,ID_EOS].item()
        if log_prob_eos > self.best_eos_sofar:
            self.log_prob_sum_since_last_best_eos = self.log_prob_sum - self.log_prob_sum_at_best_eos
            if ((self.log_prob_sum_since_last_best_eos) < (self.best_eos_sofar)) and (self.step > self.early_stop):
                return True
            else:
                self.best_eos_sofar = log_prob_eos
                self.log_prob_sum_at_best_eos = self.log_prob_sum
        self.log_prob_sum += log_prob_best
        self.step+=1
        return False


class TokenStopper:
    def __init__(self, processor, batch_size):
        self.tokenizer = processor.tokenizer
        self.eos_id = ID_EOS
        self.batch_size = batch_size
        self.eos_rows = mx.ones(batch_size)
    def __call__(self, token):
        if self.eos_id in token:
            self.eos_rows *= token.squeeze()!=self.eos_id
            if self.eos_rows.sum() < 1:
                return True
        return False


def _generate(model, processor, prompt, images=None, max_tokens=512, verbose=True, return_tps=False, early_stop=False, stream=True, mute=False):
    if images is not None and isinstance(prompt, list):
        raise ValueError('Images cannot be provided when prompt is a list')
    logit_stopper = LogitStopper(max_tokens, early_stop)
    streamer = Streamer(processor, stream, mute)
    dict_input = processor(prompt, images)
    mask, pids = dict_input.get('mask', None), dict_input.get('pids', None)
    token_stopper = TokenStopper(processor, dict_input['input_ids'].shape[0])
    tic = Tic()
    logits, cache, embedded = model(**dict_input, max_tokens=max_tokens)
    token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
    mx.eval(token, logits)
    streamer(token)
    prompt_time = tic()
    embeddings = [embedded]
    for i in range(max_tokens-1):
        logits, cache, embedded = model(input_ids=token, cache=cache, mask=mask, pids=pids)
        token = mx.argmax(logits[:, -1, :], axis=-1)[:,None]
        mx.eval(token, logits, embedded)
        embeddings += [embedded]
        streamer(token)
        if logit_stopper(logits):
            break
        if token_stopper(token):
            break
    result, gen_len = streamer.end()
    gen_time = tic()
    prompt_len = dict_input['input_ids'].size
    prompt_tps = prompt_len / prompt_time
    gen_tps = (gen_len - 1) / gen_time
    if verbose:
        print(f"\nPrompt: {prompt_tps:.2f} tokens-per-sec ({prompt_len} tokens / {prompt_time:.1f} sec)")
        print(f"Generate: {gen_tps:.2f} tokens-per-sec ({gen_len} tokens / {gen_time:.1f} sec)")
    if return_tps:
        return prompt_tps, gen_tps
    return result, embeddings
