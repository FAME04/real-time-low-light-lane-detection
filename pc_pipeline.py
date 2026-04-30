import cv2
import numpy as np

# ================================
# ENHANCEMENT FUNCTIONS
# ================================

def estimate_illumination(V):
    return cv2.GaussianBlur(V, (31, 31), 0)

def adaptive_enhance(V, illum):
    V = V.astype(np.float32)

    mean_v = np.mean(V)
    darkness_factor = 1 - (mean_v / 255.0)

    c = darkness_factor * 40  # tuned

    V_out = (V * (255 + c)) / (np.maximum(illum, V) + c)
    return np.clip(V_out, 0, 255).astype(np.uint8)

def smooth(img):
    return cv2.bilateralFilter(img, 5, 30, 30)

def sharpen(img):
    blur = cv2.GaussianBlur(img, (0,0), 1.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)

# ================================
# CAMERA
# ================================
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # ==================================
    # STEP 1: ENHANCEMENT
    # ==================================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    illum = estimate_illumination(V)
    V_enhanced = adaptive_enhance(V, illum)

    hsv_enhanced = cv2.merge([H, S, V_enhanced])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    enhanced = smooth(enhanced)
    enhanced = sharpen(enhanced)

    # ==================================
    # STEP 2: LANE DETECTION (ON ENHANCED)
    # ==================================
    lane_image = np.copy(enhanced)

    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges_gray = cv2.Canny(gray, 50, 150)
    edges_blur = cv2.Canny(blur, 50, 150)
    edges = cv2.bitwise_or(edges_gray, edges_blur)

    # ROI
    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.9 * width), height),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.4 * width), int(0.6 * height))
    ]])

    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(edges, mask)

    # HOUGH
    lines = cv2.HoughLinesP(
        cropped,
        2,
        np.pi / 180,
        80,
        minLineLength=40,
        maxLineGap=5
    )

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

    def make_coordinates(image, line_params):
        if line_params is None:
            return None

        slope, intercept = line_params
        y1 = image.shape[0]
        y2 = int(y1 * 0.6)

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return [x1, y1, x2, y2]

    left_line = make_coordinates(enhanced, left_avg)
    right_line = make_coordinates(enhanced, right_avg)

    line_img = np.zeros_like(enhanced)

    if left_line is not None:
        cv2.line(line_img,
                 (left_line[0], left_line[1]),
                 (left_line[2], left_line[3]),
                 (255, 0, 0), 8)

    if right_line is not None:
        cv2.line(line_img,
                 (right_line[0], right_line[1]),
                 (right_line[2], right_line[3]),
                 (255, 0, 0), 8)

    final_lane = cv2.addWeighted(enhanced, 0.8, line_img, 1, 1)

    # ==================================
    # DISPLAY
    # ==================================
    cv2.putText(frame, "Original", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(final_lane, "Enhanced + Lane", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    combined = np.hstack((frame, final_lane))

    cv2.imshow("Final Pipeline", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()