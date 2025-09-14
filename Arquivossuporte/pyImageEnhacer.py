from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import scipy #.ndimage as ndi  # opcional para filtros Gaussian
import matplotlib.pyplot as plt
import math



img = Image.open("Reator_Delta.jpg")
img = img.resize((img.width*2, img.height*2), Image.BICUBIC)

# --- separa alfa se existir ------------------------------------
if img.mode == "RGBA":
    rgb, alpha = img.convert("RGB"), img.getchannel("A")
else:
    rgb, alpha = img, None

# --- redução de ruído (mediana) --------------------------------
rgb = rgb.filter(ImageFilter.MedianFilter(size=3))

# --- contraste global ------------------------------------------
rgb = ImageOps.autocontrast(rgb, cutoff=2)
rgb = ImageEnhance.Contrast(rgb).enhance(1.15)

# --- correção de gama ------------------------------------------
gamma = 0.9
rgb = rgb.point(lambda p: int((p/255.0) ** (1/gamma) * 255))

# --- nitidez (unsharp mask) ------------------------------------
rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2,
                                         percent=150,
                                         threshold=3))

# === 6. brilho extra (opcional) ====
rgb = ImageEnhance.Brightness(rgb).enhance(1.1)   # +10 % brilho

# --- recoloca alfa (opcional) ----------------------------------
if alpha is not None:
    out = Image.merge("RGBA", (*rgb.split(), alpha))
else:
    out = rgb

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title("Original (ruim)")
axes[0].axis('off')

axes[1].imshow(out)
axes[1].set_title("Aprimorada (mais clara)")
axes[1].axis('off')

plt.show()

out.save("imagem_enhanced_pillow_only.png")
print("✓ Imagem salva como imagem_enhanced_pillow_only.png")
