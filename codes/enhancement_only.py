import os
import cv2
import numpy as np
import pywt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ==================================
# PATHS
# ==================================
input_folder = r"D:\Daiict\BTP\low_light_btp\dataset\night_driving"
output_folder = r"D:\Daiict\BTP\low_light_btp\dataset\night_results\adaptive_compare_updated"

# New folder for enhanced-only outputs
enhanced_folder = r"D:\Daiict\BTP\low_light_btp\dataset\night_results\enhanced_only"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(enhanced_folder, exist_ok=True)


psnr_list = []
ssim_list = []
ambe_list = []

# ==================================
# PROCESS LOOP
# ==================================
for file in os.listdir(input_folder):

    path = os.path.join(input_folder, file)
    img = cv2.imread(path)

    if img is None:
        continue

    original = img.copy()

    # 1️⃣ INPUT → HSV
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)
    V_float = V.astype(np.float32)

    # 2️⃣ GUIDED ILLUMINATION ESTIMATION
    radius = 15
    eps = 1e-3 * 255 * 255

    gf = cv2.ximgproc.createGuidedFilter(V, radius, eps)
    illum = gf.filter(V).astype(np.float32)

    # 3️⃣ DARKNESS-BASED ADAPTIVE PARAMETER
    mean_v = np.mean(V_float)

    darkness_factor = 1 - (mean_v / 255.0)

    # Scaling constant (tune between 30–60 if needed)
    c = darkness_factor * 50

    c1 = c
    c2 = c / 2

    # 4️⃣ CORRECTION (Stable Rational Model)
    V1 = (V_float * (255 + c1)) / (np.maximum(illum, V_float) + c1)
    V2 = (V_float * (255 + c2)) / (np.maximum(illum, V_float) + c2)

    V1 = np.clip(V1, 0, 255)
    V2 = np.clip(V2, 0, 255)

    # 5️⃣ WAVELET DECOMPOSITION
    coeffs1 = pywt.wavedec2(V1, "haar", level=2)
    coeffs2 = pywt.wavedec2(V2, "haar", level=2)

    fused_coeffs = []

    # 6️⃣ WEIGHTED VARIANCE FUSION
    for cA, cB in zip(coeffs1, coeffs2):

        if isinstance(cA, tuple):
            fused_subbands = []
            for bandA, bandB in zip(cA, cB):
                varA = np.var(bandA)
                varB = np.var(bandB)

                eps_small = 1e-6
                wA = varA / (varA + varB + eps_small)
                wB = varB / (varA + varB + eps_small)

                fused_subbands.append(wA * bandA + wB * bandB)

            fused_coeffs.append(tuple(fused_subbands))

        else:
            varA = np.var(cA)
            varB = np.var(cB)

            eps_small = 1e-6
            wA = varA / (varA + varB + eps_small)
            wB = varB / (varA + varB + eps_small)

            fused_coeffs.append(wA * cA + wB * cB)

    # 7️⃣ INVERSE DWT
    V_fused = pywt.waverec2(fused_coeffs, "haar")
    V_fused = np.clip(V_fused, 0, 255)

    h, w = V.shape
    V_fused = V_fused[:h, :w].astype(np.uint8)

    hsv_fused = cv2.merge([H, S, V_fused])
    final = cv2.cvtColor(hsv_fused, cv2.COLOR_HSV2BGR)

    # 8️⃣ METRICS
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

    psnr_list.append(peak_signal_noise_ratio(orig_gray, enhanced_gray))
    ssim_list.append(structural_similarity(orig_gray, enhanced_gray))
    ambe_list.append(abs(np.mean(orig_gray) - np.mean(enhanced_gray)))

    # SAVE SIDE-BY-SIDE
    compare = np.hstack((original, final))
    cv2.imwrite(os.path.join(output_folder, file), compare)

    # Save enhanced-only image
    cv2.imwrite(os.path.join(enhanced_folder, file), final)

    # Save side-by-side comparison
    compare = np.hstack((original, final))
    cv2.imwrite(os.path.join(output_folder, file), compare)


# ==================================
# FINAL RESULTS
# ==================================
print("\n===== UPDATED ADAPTIVE VERSION =====")
print("Average PSNR :", np.mean(psnr_list))
print("Average SSIM :", np.mean(ssim_list))
print("Average AMBE :", np.mean(ambe_list))
print("\nAdaptive updated processing completed.")

