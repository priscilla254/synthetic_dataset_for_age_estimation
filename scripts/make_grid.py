#!/usr/bin/env python3
"""
make_image_grid.py
Create a grid montage from images in a folder and save it in that folder.
Now supports per-image filename labels.

Dependencies: Pillow (pip install pillow)

Examples:
  python make_image_grid.py "/path/to/folder" --labels
  python make_image_grid.py "/path/to/folder" --labels --label-size 16 --label-bg "#111" --label-color "#fff"
  python make_image_grid.py "/path/to/folder" --labels --font "/System/Library/Fonts/SFNSMono.ttf"
"""

import argparse
import math
import random
from pathlib import Path
from typing import Tuple, List, Optional

from PIL import Image, ImageOps, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_OUT = "grid.jpg"

def parse_cell(s: str) -> Tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Cell must look like WIDTHxHEIGHT, e.g., 256x256")

def parse_bg(s: str) -> Tuple[int, int, int]:
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(c*2 for c in s)
    if len(s) != 6:
        raise argparse.ArgumentTypeError("Color must be hex like #ffffff or #111")
    try:
        return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        raise argparse.ArgumentTypeError("Color must be hex like #ffffff")

def find_images(folder: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    files.sort(key=lambda p: p.name.lower())
    return files

def load_font(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    # Fallback to default bitmap font
    return ImageFont.load_default()

def ellipsize_to_fit(draw: ImageDraw.ImageDraw, text: str, max_width: int, font: ImageFont.ImageFont) -> str:
    # Quick path: fits as-is
    if draw.textlength(text, font=font) <= max_width:
        return text
    # Try progressively chopping and adding ellipsis
    ell = "…"
    if draw.textlength(ell, font=font) > max_width:
        return ""  # hopelessly small
    lo, hi = 0, len(text)
    ans = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + ell
        if draw.textlength(candidate, font=font) <= max_width:
            ans = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return ans

def letterbox_fit(im: Image.Image, target_size: Tuple[int, int], bg: Tuple[int, int, int]) -> Image.Image:
    """Resize image to fit within target_size preserving aspect; pad with bg color."""
    tw, th = target_size
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    iw, ih = im.size
    if iw == 0 or ih == 0:
        return Image.new("RGB", (tw, th), bg)
    scale = min(tw / iw, th / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    im_resized = im.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), bg)
    ox = (tw - nw) // 2
    oy = (th - nh) // 2
    canvas.paste(im_resized, (ox, oy))
    return canvas

def make_tiled_with_label(
    tile_img: Image.Image,
    label_text: str,
    cell_size: Tuple[int, int],
    show_label: bool,
    font: ImageFont.ImageFont,
    label_h: int,
    label_bg: Tuple[int, int, int],
    label_fg: Tuple[int, int, int],
    bg: Tuple[int, int, int],
    label_padding: int = 4,
) -> Image.Image:
    """Return a tile possibly augmented with a label strip at the bottom."""
    cw, ch = cell_size
    if not show_label or label_h <= 0:
        # Just ensure tile matches cell size
        if tile_img.size != (cw, ch):
            tile_img = tile_img.resize((cw, ch), Image.LANCZOS)
        return tile_img

    # Create combined tile: image area (ch - label_h) + label strip
    img_area_h = max(1, ch - label_h)
    # First, resize content to fit IMG area (maintain aspect via letterbox inside that sub-rect)
    content = tile_img
    if content.size != (cw, img_area_h):
        # letterbox into cw x img_area_h
        content = letterbox_fit(tile_img, (cw, img_area_h), bg)

    # Compose final tile
    out = Image.new("RGB", (cw, ch), bg)
    out.paste(content, (0, 0))

    # Draw label strip
    draw = ImageDraw.Draw(out)
    y0 = ch - label_h
    draw.rectangle([0, y0, cw, ch], fill=label_bg)

    # Pad inside strip
    max_text_w = max(1, cw - 2 * label_padding)
    text = ellipsize_to_fit(draw, label_text, max_text_w, font)
    # Vertical centering
    tb = draw.textbbox((0, 0), text, font=font)
    tw = tb[2] - tb[0]
    th = tb[3] - tb[1]
    tx = label_padding
    ty = y0 + (label_h - th) // 2  # vertically centered within strip
    draw.text((tx, ty), text, font=font, fill=label_fg)
    return out

def auto_cols(n: int) -> int:
    c = math.ceil(math.sqrt(n))
    while c > 1 and (math.ceil(n / c) - 1) * c >= n:
        c -= 1
    return c

def build_grid(
    images: List[Path],
    out_path: Path,
    cols: int,
    cell: Tuple[int, int],
    pad: int,
    margin: int,
    bg: Tuple[int, int, int],
    labels: bool,
    label_size: int,
    label_bg: Tuple[int, int, int],
    label_fg: Tuple[int, int, int],
    font_path: Optional[str],
):
    if not images:
        raise RuntimeError("No images found to build a grid.")

    n = len(images)
    if cols <= 0:
        cols = auto_cols(n)
    rows = math.ceil(n / cols)

    cell_w, cell_h = cell
    grid_w = cols * cell_w + (cols - 1) * pad + 2 * margin
    grid_h = rows * cell_h + (rows - 1) * pad + 2 * margin

    montage = Image.new("RGB", (grid_w, grid_h), bg)

    # Prepare font + label strip height (roughly ~1.6x font size)
    font = load_font(font_path, label_size)
    label_h = int(round(label_size * 1.6)) if labels else 0

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            x = margin + c * (cell_w + pad)
            y = margin + r * (cell_h + pad)
            p = images[idx]
            try:
                with Image.open(p) as im:
                    # First make image fit full cell, then overlay label area (which internally re-fits content)
                    base_tile = letterbox_fit(im, (cell_w, cell_h), bg)
            except Exception:
                base_tile = Image.new("RGB", (cell_w, cell_h), bg)

            label_text = p.stem  # filename without extension
            tile = make_tiled_with_label(
                base_tile,
                label_text,
                (cell_w, cell_h),
                labels,
                font,
                label_h,
                label_bg,
                label_fg,
                bg,
            )
            montage.paste(tile, (x, y))
            idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        montage.save(out_path, quality=92, subsampling=1, optimize=True)
    elif ext in {".png"}:
        montage.save(out_path, optimize=True)
    else:
        out_path = out_path.with_suffix(".jpg")
        montage.save(out_path, quality=92, subsampling=1, optimize=True)
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Create an image grid from a folder of images.")
    ap.add_argument("folder", type=str, help="Folder containing images.")
    ap.add_argument("--out", type=str, default=DEFAULT_OUT,
                    help=f"Output file name (saved inside the folder). Default: {DEFAULT_OUT}")
    ap.add_argument("--cols", type=int, default=0, help="Number of columns (0 = auto).")
    ap.add_argument("--cell", type=parse_cell, default=(256, 256),
                    help="Cell size as WIDTHxHEIGHT, e.g., 256x256. Default: 256x256")
    ap.add_argument("--pad", type=int, default=6, help="Padding between cells (pixels). Default: 6")
    ap.add_argument("--margin", type=int, default=12, help="Outer margin (pixels). Default: 12")
    ap.add_argument("--bg", type=parse_bg, default=parse_bg("#ffffff"),
                    help='Background color as hex, e.g., "#111111" or "#fff". Default: #ffffff')
    ap.add_argument("--recursive", action="store_true", help="Include images in subfolders.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle image order.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = no limit).")

    # Label options
    ap.add_argument("--labels", action="store_true", help="Draw each image's base filename under the tile.")
    ap.add_argument("--label-size", type=int, default=14, help="Font size for labels. Default: 14")
    ap.add_argument("--label-bg", type=parse_bg, default=parse_bg("#000000"),
                    help='Label strip background (hex). Default: #000000')
    ap.add_argument("--label-color", type=parse_bg, default=parse_bg("#ffffff"),
                    help='Label text color (hex). Default: #ffffff')
    ap.add_argument("--font", type=str, default=None,
                    help="Path to a .ttf/.otf font. Defaults to Pillow's built-in font.")

    args = ap.parse_args()
    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    images = find_images(folder, recursive=args.recursive)
    if not images:
        raise SystemExit("No image files found.")

    if args.shuffle:
        random.seed(42)
        random.shuffle(images)

    if args.limit and args.limit > 0:
        images = images[: args.limit]

    out_path = (folder / args.out)
    final_path = build_grid(
        images=images,
        out_path=out_path,
        cols=args.cols,
        cell=args.cell,
        pad=args.pad,
        margin=args.margin,
        bg=args.bg,
        labels=args.labels,
        label_size=args.label_size,
        label_bg=args.label_bg,
        label_fg=args.label_color,
        font_path=args.font,
    )
    print(f"Saved grid to: {final_path}")

if __name__ == "__main__":
    main()
