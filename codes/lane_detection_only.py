import cv2
import numpy as np
import os

# ===== PATHS =====
input_folder = r"D:\Daiict\BTP\low_light_btp\src\Road_Lane_Detection\Input"
debug_folder = r"D:\Daiict\BTP\low_light_btp\src\Road_Lane_Detection\Sample_Output_Multiple"
final_output_folder = r"D:\Daiict\BTP\low_light_btp\src\Road_Lane_Detection\Output4"

os.makedirs(debug_folder, exist_ok=True)
os.makedirs(final_output_folder, exist_ok=True)

# ===== LOOP THROUGH ALL IMAGES =====
for filename in os.listdir(input_folder):

    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    print(f"Processing: {filename}")

    input_path = os.path.join(input_folder, filename)
    image = cv2.imread(input_path)

    if image is None:
        print(f"❌ Skipping {filename}")
        continue

    lane_image = np.copy(image)

    # ===== 1. GRAYSCALE =====
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)

    # ===== 2. GAUSSIAN BLUR =====
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ===== 3. CANNY EDGES (DUAL + OR) =====
    edges_gray = cv2.Canny(gray, 50, 150)
    edges_blur = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_or(edges_gray, edges_blur)

    # ===== 4. ROI MASK (YOUR FIXED TRAPEZIUM) =====
    height, width = edges.shape

    mask = np.zeros_like(edges)

    polygon = np.array([[
        (200, height),     # bottom left
        (1200, height),    # bottom right
        (1000, 250),       # top right
        (250, 250)         # top left
    ]])

    cv2.fillPoly(mask, polygon, 255)

    # ===== 5. ROI APPLIED =====
    cropped = cv2.bitwise_and(edges, mask)

    # ===== ROI VISUAL =====
    roi_visual = lane_image.copy()
    cv2.polylines(roi_visual, polygon, isClosed=True, color=(0, 255, 0), thickness=3)

    # ===== 6. HOUGH LINES =====
    lines = cv2.HoughLinesP(
        cropped,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=5
    )

    line_debug = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ===== 7. FILTER + AVERAGE =====
    left_fit = []
    right_fit = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length < 50:
                continue

            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_avg = np.mean(left_fit, axis=0) if len(left_fit) > 0 else None
    right_avg = np.mean(right_fit, axis=0) if len(right_fit) > 0 else None

    # ===== 8. FINAL LANE LINES =====
    def make_coordinates(image, line_params):
        if line_params is None:
            return None

        slope, intercept = line_params
        y1 = image.shape[0]
        y2 = int(y1 * 0.6)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return [x1, y1, x2, y2]

    left_line = make_coordinates(image, left_avg)
    right_line = make_coordinates(image, right_avg)

    final_lines_img = np.zeros_like(image)

    if left_line is not None:
        cv2.line(final_lines_img,
                 (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]),
                 (255, 0, 0), 10)

    if right_line is not None:
        cv2.line(final_lines_img,
                 (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]),
                 (255, 0, 0), 10)

    # ===== 9. FINAL OVERLAY =====
    final_image = cv2.addWeighted(image, 0.8, final_lines_img, 1, 1)

    # ===== SAVE FILES =====
    name = os.path.splitext(filename)[0]

    # Debug images
    cv2.imwrite(os.path.join(debug_folder, f"{name}_edges.jpg"), edges)
    cv2.imwrite(os.path.join(debug_folder, f"{name}_roi.jpg"), roi_visual)
    cv2.imwrite(os.path.join(debug_folder, f"{name}_hough.jpg"), line_debug)

    # Final output only
    cv2.imwrite(os.path.join(final_output_folder, f"{name}.jpg"), final_image)

print("✅ All images processed and saved!")