import cv2
import numpy as np

# --- Carregar imagem original (a menos nítida) ------------------------------
img = cv2.imread("Reator_Delta.jpg")

# # A) ↑ Upscale 2× (interpolação cúbica) – opcional
# img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#
# # B) ↓ Denoise fino preservando bordas
# den = cv2.fastNlMeansDenoisingColored(img, None,
#                                       h        = 10,  # força luminância
#                                       hColor   = 10,  # força cor
#                                       templateWindowSize = 7,
#                                       searchWindowSize   = 21)
#
# # C) ↑ CLAHE no canal L de LAB
# lab   = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
# l, a, b = cv2.split(lab)
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# l2    = clahe.apply(l)
# lab   = cv2.merge((l2, a, b))
# clahe_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#
# # D) Ajuste gama (levemente mais claro)
# gamma = 1.1
# # table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
# #                   for i in np.arange(0, 256)]).astype("uint8")
# # g_out = cv2.LUT(clahe_out, table)
# #
# # # E) Unsharp Mask (high-boost)
# # blur   = cv2.GaussianBlur(g_out, (0, 0), sigmaX=3)
# # sharp  = cv2.addWeighted(g_out, 1.5, blur, -0.5, 0)
# #
# # # F) Normalizar e salvar
# # final = cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX)
# # cv2.imwrite("imagem_enhanced_ocv.png", final)
