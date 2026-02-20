import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

# ===============================
# LOAD YOUR POINTS
# ===============================
points = np.array([
[18,8],[19,9],[20,9],[9,10],[10,10],[21,10],[22,10],
[9,11],[11,11],[19,11],[20,11],[22,11],
[9,12],[11,12],[20,12],[22,12],
[9,13],[10,13],[11,13],[20,13],[23,13],
[9,14],[11,14],[21,14],[22,14],[23,14],
[9,15],[11,15],[21,15],
[9,16],[11,16],[22,16],[24,16],
[0,17],[9,17],[12,17],[22,17],[24,17],
[0,18],[12,18],[21,18],
[0,19],[10,19],[12,19],[20,19],
[10,20],[13,20],[14,20],[20,20],[22,20],
[10,21],[11,21],[12,21],[14,21],[17,21],
[18,21],[19,21],[22,21],[30,21],[31,21],
[12,22],[13,22],[14,22],[17,22],[21,22],
[25,22],[26,22],[27,22],[32,22],
[17,23],[18,23],[19,23],[20,23],[21,23],
[28,23],[30,23],[35,23],[36,23],[37,23],
[23,24],[25,24],[26,24],[27,24],[28,24],
[21,25],[22,25],[23,25],[34,25],[35,25],
[36,25],[37,25],
[23,26],[38,26],
[20,27],[23,27],[36,27],[38,27],
[20,28],[22,28],[36,28],
[37,29],[38,29],[39,29],
[18,30],[19,30],[20,30],[37,30],[39,30],
[18,31],[20,31],[37,31],[39,31],
[18,32],[19,32],[20,32],[37,32],[39,32],
[18,33],[37,33],
[18,34],[20,34],[36,34],
[18,35],[20,35],[36,35],
[18,36],[19,36],[20,36],[34,36],[35,36],[37,36],
[18,37],[21,37],[34,37],[37,37],
[18,38],[22,38],[35,38],[36,38],
[19,39],[32,39],[33,39],
[20,40],[24,40],[30,40],[31,40],[34,40],
[22,41],[26,41],[27,41],[29,41],[33,41],[34,41],
[23,42],[24,42],[27,42],[30,42],[31,42],
[25,43],[26,43],[27,43]
])

# ===============================
# PARAMETERS
# ===============================
NUM_RANDOM_POINTS = 20
MAX_ITER = 2000
EXTREME_THRESHOLD_RATIO = 0.15   # 15% tolerance

# ===============================
# EXTREME CHECK FUNCTION
# ===============================
def extreme_check(pts):
    x = pts[:, 0]
    y = pts[:, 1]

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    width = max_x - min_x
    height = max_y - min_y

    left_zone = min_x + width * EXTREME_THRESHOLD_RATIO
    right_zone = max_x - width * EXTREME_THRESHOLD_RATIO
    top_zone = min_y + height * EXTREME_THRESHOLD_RATIO
    bottom_zone = max_y - height * EXTREME_THRESHOLD_RATIO

    has_left = np.any(x <= left_zone)
    has_right = np.any(x >= right_zone)
    has_top = np.any(y <= top_zone)
    has_bottom = np.any(y >= bottom_zone)

    return has_left and has_right and has_top and has_bottom

# ===============================
# FIT ELLIPSE
# ===============================
def fit_ellipse(pts):
    try:
        ellipse = cv2.fitEllipse(pts.astype(np.float32))
        return ellipse
    except:
        return None

# ===============================
# RANDOM SEARCH WITH EXTREME VALIDATION
# ===============================
best_ellipse = None

for i in range(MAX_ITER):

    idx = random.sample(range(len(points)), NUM_RANDOM_POINTS)
    sample_pts = points[idx]

    # STEP 1: Check extreme coverage
    if not extreme_check(sample_pts):
        continue

    # STEP 2: Fit ellipse
    ellipse = fit_ellipse(sample_pts)
    if ellipse is None:
        continue

    # STEP 3: Accept
    best_ellipse = ellipse
    print("Valid ellipse found at iteration:", i)
    break

# ===============================
# VISUALIZE RESULT
# ===============================
if best_ellipse is not None:
    canvas = np.zeros((int(np.max(points[:,1])+20),
                       int(np.max(points[:,0])+20), 3), dtype=np.uint8)

    for p in points:
        canvas[int(p[1]), int(p[0])] = (255,255,255)

    cv2.ellipse(canvas, best_ellipse, (0,0,255), 1)

    plt.imshow(canvas)
    plt.title("Detected Ellipse (Extreme Validated)")
    plt.gca().invert_yaxis()
    plt.show()

else:
    print("No ellipse found.")
