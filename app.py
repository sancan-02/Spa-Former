"""
Spa-Former Image Inpainting — Gradio UI
Run: python app.py
"""

import os
import random
from random import randint
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "generator.pt")
IMG_SIZE = 256

# ---------------------------------------------------------------------------
# Mask functions (from model.ipynb) — 1 = known, 0 = hole
# ---------------------------------------------------------------------------

def random_regular_mask(img):
    s = img.size()
    mask = torch.ones(1, s[1], s[2])
    n_mask = random.randint(1, 5)
    limx = s[1] - s[1] / (n_mask + 1)
    limy = s[2] - s[2] / (n_mask + 1)
    for _ in range(n_mask):
        x = random.randint(0, int(limx))
        y = random.randint(0, int(limy))
        range_x = x + random.randint(int(s[1] / (n_mask + 7)), int(s[1] - x))
        range_y = y + random.randint(int(s[2] / (n_mask + 7)), int(s[2] - y))
        mask[:, int(x):int(range_x), int(y):int(range_y)] = 0
    return mask


def center_mask(img):
    size = img.size()
    mask = torch.ones(1, size[1], size[2])
    x, y = int(size[1] / 4), int(size[2] / 4)
    mask[:, x:int(size[1] * 3 / 4), y:int(size[2] * 3 / 4)] = 0
    return mask


def random_irregular_mask(img):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    size = img.size()
    mask = torch.ones(1, size[1], size[2])
    canvas = np.zeros((size[1], size[2], 1), np.uint8)
    max_width = 20
    for _ in range(random.randint(16, 64)):
        mode = random.random()
        if mode < 0.6:
            x1, x2 = randint(1, size[1]), randint(1, size[1])
            y1, y2 = randint(1, size[2]), randint(1, size[2])
            cv2.line(canvas, (x1, y1), (x2, y2), (1, 1, 1), randint(4, max_width))
        elif mode < 0.8:
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            cv2.circle(canvas, (x1, y1), randint(4, max_width), (1, 1, 1), -1)
        else:
            x1, y1 = randint(1, size[1]), randint(1, size[2])
            s1, s2 = randint(1, size[1]), randint(1, size[2])
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            cv2.ellipse(canvas, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), randint(4, max_width))
    canvas = canvas.reshape(size[2], size[1])
    img_mask = to_tensor(Image.fromarray(canvas * 255))
    mask[0, :, :] = img_mask < 1
    return mask


def random_freefrom_mask(img, mv=5, ma=4.0, ml=40, mbw=10):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    size = img.size()
    mask = torch.ones(1, size[1], size[2])
    canvas = np.zeros((size[1], size[2], 1), np.uint8)
    num_v = 12 + np.random.randint(mv)
    for i in range(num_v):
        start_x = np.random.randint(size[1])
        start_y = np.random.randint(size[2])
        for _ in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(int(ma))
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(ml)
            brush_w = 10 + np.random.randint(mbw)
            end_x = int(start_x + length * np.sin(angle))
            end_y = int(start_y + length * np.cos(angle))
            cv2.line(canvas, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y
    canvas = canvas.reshape(size[2], size[1])
    img_mask = to_tensor(Image.fromarray(canvas * 255))
    mask[0, :, :] = img_mask < 1
    return mask


MASK_FNS = {
    "Center": center_mask,
    "Random Rectangular": random_regular_mask,
    "Random Irregular": random_irregular_mask,
    "Free-form Brush": random_freefrom_mask,
}

# ---------------------------------------------------------------------------
# Model Architecture (from model.ipynb)
# ---------------------------------------------------------------------------

class ChannelLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gn = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.gn(x)


class SpaAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.norm1 = ChannelLayerNorm(dim)
        self.gate = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=True), nn.GELU())
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        x_n = self.norm1(x)
        g = self.gate(x_n)
        if mask is not None:
            mask_resized = F.interpolate(mask, size=(h, w), mode="nearest")
            g = g * mask_resized
        qkv = self.qkv_dwconv(self.qkv(x_n))
        q, k, v = qkv.chunk(3, dim=1)
        c_per_head = c // self.num_heads
        n = h * w
        q = q.view(b, self.num_heads, c_per_head, n).permute(0, 1, 3, 2)
        k = k.view(b, self.num_heads, c_per_head, n).permute(0, 1, 3, 2)
        v = v.view(b, self.num_heads, c_per_head, n).permute(0, 1, 3, 2)
        q = F.normalize(q, dim=2)
        k = F.normalize(k, dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = F.relu(attn)
        out = v @ attn
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = out * g
        return self.project_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.norm = ChannelLayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=False),
            nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, stride=1, padding=1, groups=hidden * 2, bias=False),
        )
        self.linear = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x_n = self.norm(x)
        x1, x2 = self.conv(x_n).chunk(2, dim=1)
        return self.linear(F.gelu(x1) * x2)


class TransformerBlock(nn.Module):
    def __init__(self, in_ch, head, expansion_factor=2.66):
        super().__init__()
        self.attn = SpaAttention(dim=in_ch, num_heads=head)
        self.ffn = FeedForward(dim=in_ch, expansion_factor=expansion_factor)

    def forward(self, x, mask=None):
        x = x + self.attn(x, mask=mask)
        x = x + self.ffn(x)
        return x


class Downsample(nn.Module):
    def __init__(self, num_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_ch, num_ch * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_ch * 2, track_running_stats=False),
            nn.GELU(),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, num_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_ch, num_ch // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(num_ch // 2, track_running_stats=False),
            nn.GELU(),
        )

    def forward(self, x):
        return self.body(F.interpolate(x, scale_factor=2, mode="nearest"))


class Encoder(nn.Module):
    def __init__(self, ngf=48, num_blocks=(1, 2, 3, 4), num_heads=(1, 2, 4, 8), expansion_factor=2.66):
        super().__init__()
        self.start = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.GELU(),
        )
        self.trane256 = nn.Sequential(*[TransformerBlock(ngf, num_heads[0], expansion_factor) for _ in range(num_blocks[0])])
        self.down128 = Downsample(ngf)
        self.trane128 = nn.Sequential(*[TransformerBlock(ngf * 2, num_heads[1], expansion_factor) for _ in range(num_blocks[1])])
        self.down64 = Downsample(ngf * 2)
        self.trane64 = nn.Sequential(*[TransformerBlock(ngf * 4, num_heads[2], expansion_factor) for _ in range(num_blocks[2])])
        self.down32 = Downsample(ngf * 4)
        self.trane32 = nn.Sequential(*[TransformerBlock(ngf * 8, num_heads[3], expansion_factor) for _ in range(num_blocks[3])])

    def forward(self, x, mask=None, return_pyramid=False):
        if mask is None:
            mask = torch.ones(x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        feat = torch.cat([x, mask], dim=1)
        f256 = self.trane256(self.start(feat))
        f128 = self.trane128(self.down128(f256))
        f64 = self.trane64(self.down64(f128))
        f32 = self.trane32(self.down32(f64))
        if return_pyramid:
            return f256, f128, f64, f32
        return f32


class Decoder(nn.Module):
    def __init__(self, ngf=48, num_blocks=(1, 2, 3, 4), num_heads=(1, 2, 4, 8), expansion_factor=2.66):
        super().__init__()
        self.up64 = Upsample(ngf * 8)
        self.fuse64 = nn.Conv2d(ngf * 8, ngf * 4, kernel_size=1, bias=False)
        self.trand64 = nn.Sequential(*[TransformerBlock(ngf * 4, num_heads[2], expansion_factor) for _ in range(num_blocks[2])])
        self.up128 = Upsample(ngf * 4)
        self.fuse128 = nn.Conv2d(ngf * 4, ngf * 2, kernel_size=1, bias=False)
        self.trand128 = nn.Sequential(*[TransformerBlock(ngf * 2, num_heads[1], expansion_factor) for _ in range(num_blocks[1])])
        self.up256 = Upsample(ngf * 2)
        self.fuse256 = nn.Conv2d(ngf * 2, ngf, kernel_size=1, bias=False)
        self.trand256 = nn.Sequential(*[TransformerBlock(ngf, num_heads[0], expansion_factor) for _ in range(num_blocks[0])])
        self.out = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, 3, kernel_size=7, padding=0))

    def forward(self, f256, f128, f64, f32):
        out64 = self.fuse64(torch.cat([f64, self.up64(f32)], dim=1))
        out64 = self.trand64(out64)
        out128 = self.fuse128(torch.cat([f128, self.up128(out64)], dim=1))
        out128 = self.trand128(out128)
        out256 = self.fuse256(torch.cat([f256, self.up256(out128)], dim=1))
        out256 = self.trand256(out256)
        return torch.tanh(self.out(out256))


class Generator(nn.Module):
    def __init__(self, ngf=48, num_blocks=(1, 2, 3, 4), num_heads=(1, 2, 4, 8), expansion_factor=2.66, use_input_noise=True):
        super().__init__()
        self.use_input_noise = use_input_noise
        self.encoder = Encoder(ngf=ngf, num_blocks=num_blocks, num_heads=num_heads, expansion_factor=expansion_factor)
        self.decoder = Decoder(ngf=ngf, num_blocks=num_blocks, num_heads=num_heads, expansion_factor=expansion_factor)

    def forward(self, x, mask):
        if self.use_input_noise:
            x = x + torch.randn_like(x) * (1.0 / 128.0)
        f256, f128, f64, f32 = self.encoder(x, mask, return_pyramid=True)
        return self.decoder(f256, f128, f64, f32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

G = None

def _load_model():
    global G
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Weights not found at '{WEIGHTS_PATH}'.\n"
            "Place your trained .pt file at: checkpoints/generator.pt"
        )
    G = Generator(use_input_noise=False)
    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    # Support multiple checkpoint formats
    if isinstance(ckpt, dict):
        state = ckpt.get("G") or ckpt.get("generator") or ckpt.get("state_dict") or ckpt
    else:
        state = ckpt
    G.load_state_dict(state)
    G.eval().to(device)
    print(f"Loaded weights from {WEIGHTS_PATH} | device: {device}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def _run_inpainting(image, mask_type):
    if G is None:
        raise gr.Error("No weights loaded. Add checkpoints/generator.pt and restart the app.")
    if image is None:
        raise gr.Error("Please upload an image first.")

    original = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_np = np.array(original).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    mask = MASK_FNS[mask_type](img_tensor.squeeze(0)).unsqueeze(0).to(device)
    img_masked = img_tensor * mask

    with torch.no_grad():
        pred = G(img_masked, mask).clamp(0, 1)
        completed = pred * (1.0 - mask) + img_masked * mask

    return _tensor_to_pil(img_masked), _tensor_to_pil(pred), _tensor_to_pil(completed)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0f0f0f",
        body_background_fill_dark="#0f0f0f",
        block_background_fill="#1a1a1a",
        block_background_fill_dark="#1a1a1a",
        block_border_color="#3730a3",
        block_border_color_dark="#3730a3",
        block_label_text_color="#888888",
        block_label_text_color_dark="#888888",
        block_title_text_color="#cccccc",
        block_title_text_color_dark="#cccccc",
        body_text_color="#cccccc",
        body_text_color_dark="#cccccc",
        body_text_color_subdued="#666666",
        button_primary_background_fill="#6366f1",
        button_primary_background_fill_dark="#6366f1",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_background_fill_hover="#4f46e5",
        button_primary_background_fill_hover_dark="#4f46e5",
        input_background_fill="#1a1a1a",
        input_background_fill_dark="#1a1a1a",
        input_border_color="#2a2a2a",
        input_border_color_dark="#2a2a2a",
        shadow_drop="none",
        shadow_drop_lg="none",
        block_radius="8px",
        button_large_radius="6px",
    )

    css = """
    .gradio-container { max-width: 98vw !important; margin: 32px auto; padding: 0 24px; }
    footer { display: none !important; }
    #title { margin-bottom: 4px; }
    #title h1 { font-size: 2rem !important; font-weight: 600 !important; color: #a5b4fc !important; margin: 0; }
    #subtitle p { color: #555555 !important; font-size: 0.8rem !important; margin: 0 0 24px 0; }
    #run_btn { margin-top: 12px; }
    .block { border-radius: 8px !important; }
    .gap { gap: 12px !important; }
    .gr-row { flex-wrap: nowrap !important; }
    .image-editor-toolbar { display: none !important; }
    [data-testid="image-editor"] .toolbar { display: none !important; }
    """

    with gr.Blocks(theme=theme, title="Spa-Former", css=css) as demo:

        gr.Markdown("# Spa-Former · Inpainting", elem_id="title")
        gr.Markdown("Upload an image, choose a mask type, click Inpaint.", elem_id="subtitle")

        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                input_image = gr.Image(label="Input", type="pil", height=320)
                mask_dropdown = gr.Dropdown(
                    choices=list(MASK_FNS.keys()),
                    value="Random Rectangular",
                    label="Mask Type",
                )
                run_btn = gr.Button("Inpaint", variant="primary", size="lg", elem_id="run_btn")

            with gr.Column(scale=3):
                out_masked = gr.Image(label="Masked", type="pil", height=360, interactive=False)

            with gr.Column(scale=3):
                out_raw = gr.Image(label="Output", type="pil", height=360, interactive=False)

            with gr.Column(scale=3):
                out_composite = gr.Image(label="Result", type="pil", height=360, interactive=False)

        run_btn.click(
            fn=_run_inpainting,
            inputs=[input_image, mask_dropdown],
            outputs=[out_masked, out_raw, out_composite],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        _load_model()
    except FileNotFoundError as e:
        print(f"\n[WARNING] {e}\nStarting UI in preview mode — inference will be disabled.\n")
    ui = build_ui()
    ui.launch(share=False)
