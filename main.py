# Requires: pip install mediapipe
import cv2
import numpy as np
import random
import os
import mediapipe as mp

# --- Global Variables ---
# Drawing modes
current_mode = "pen"  # "pen" or "eraser"

# Colors (BGR format for OpenCV drawing)
PEN_COLOR_GREEN_BGR = (0, 255, 0)    # Green for pen
ERASER_COLOR_WHITE_BGR = (255, 255, 255) # White for eraser (draws white on canvas)
BUTTON_TEXT_COLOR_DARK = (0,0,0)
BUTTON_TEXT_COLOR_LIGHT = (255,255,255)
HIGHLIGHT_COLOR = (255, 255, 0) # Cyan for highlighting active button or detected object
CLEAR_BUTTON_COLOR = (50, 50, 200) # Reddish for clear button
PEN_BUTTON_COLOR = (0, 200, 0) # Green for pen button
ERASER_BUTTON_COLOR = (200, 200, 200) # Light gray for eraser button
FISH_BACKGROUND_COLOR = (255, 220, 180) # Light blue for water background

# Thickness
PEN_THICKNESS = 5
ERASER_THICKNESS = 50 # Radius of the eraser circle

# HSV Color Ranges for Detection (Tune these for your objects and lighting)
GREEN_LOWER = np.array([35, 90, 90])
GREEN_UPPER = np.array([85, 255, 255])
WHITE_LOWER = np.array([0, 0, 160])
WHITE_UPPER = np.array([180, 80, 255])
SKIN_LOWER = np.array([0, 30, 60])
SKIN_UPPER = np.array([20, 150, 255])

# Canvas for drawing
canvas = None
last_points = {"pen_green": None}

# Button Definitions
BUTTON_MARGIN = 20
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 50
BUTTON_PEN_RECT = (BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT)
BUTTON_ERASER_RECT = (BUTTON_MARGIN, BUTTON_MARGIN * 2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT)
BUTTON_CLEAR_RECT = (BUTTON_MARGIN, BUTTON_MARGIN * 3 + BUTTON_HEIGHT * 2, BUTTON_WIDTH, BUTTON_HEIGHT)

# --- Fish Animation ---
NUM_FISH = 15
FISH_COLORS = [
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (0, 165, 255),  # Orange
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255,192,203),  # Pink
    (0,128,0),      # Dark Green
    (0,255,0),      # Bright Green (new)
    (255,255,255),  # White (new)
    (0,140,255)     # Deep Orange (new)
]
fish_list = []

# --- PNG Fish Image Loading ---
FISH_PNG_PATHS = ['fish.png', 'fish2.png', 'fish3.png', 'fish4.png']
fish_png_imgs = []
for path in FISH_PNG_PATHS:
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            fish_png_imgs.append(img)

class Fish:
    """Represents a single animated fish."""
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.x = random.randint(0, frame_width)
        self.y = random.randint(0, frame_height)
        self.size = random.randint(25, 50) * 2  # Body length, doubled for bigger fish
        self.color = random.choice(FISH_COLORS)
        self.dx = random.choice([-2, -1.5, -1, 1, 1.5, 2])  # Speed in x
        self.dy = random.choice([-1, -0.5, 0.5, 1])      # Speed in y
        # Randomly choose to use one of the PNGs (if available)
        self.png_idx = None
        if fish_png_imgs and random.random() < 0.5:
            self.png_idx = random.randint(0, len(fish_png_imgs)-1)

    def update(self):
        self.x += self.dx
        self.y += self.dy
        # Boundary checks and direction change
        if self.x - self.size // 2 < 0:
            self.x = self.size // 2
            self.dx *= -1
        elif self.x + self.size // 2 > self.frame_width:
            self.x = self.frame_width - self.size // 2
            self.dx *= -1
        if self.y - self.size // 4 < 0:
            self.y = self.size // 4
            self.dy *= -1
        elif self.y + self.size // 4 > self.frame_height:
            self.y = self.frame_height - self.size // 4
            self.dy *= -1

    def draw(self, on_frame):
        body_center = (int(self.x), int(self.y))
        body_axes = (self.size // 2, self.size // 4)
        if self.png_idx is not None and 0 <= self.png_idx < len(fish_png_imgs):
            overlay_png(on_frame, fish_png_imgs[self.png_idx], body_center[0], body_center[1], size=(self.size, self.size))
        else:
            # Old OpenCV drawing
            cv2.ellipse(on_frame, body_center, body_axes, 0, 0, 360, self.color, -1)
            # Tail (triangle)
            tail_size = self.size // 3
            if self.dx > 0:  # Moving right, tail on the left
                pt1 = (body_center[0] - body_axes[0], body_center[1])
                pt2 = (pt1[0] - tail_size, pt1[1] - tail_size)
                pt3 = (pt1[0] - tail_size, pt1[1] + tail_size)
            else:  # Moving left (or stationary), tail on the right
                pt1 = (body_center[0] + body_axes[0], body_center[1])
                pt2 = (pt1[0] + tail_size, pt1[1] - tail_size)
                pt3 = (pt1[0] + tail_size, pt1[1] + tail_size)
            triangle_pts = np.array([pt1, pt2, pt3], dtype=np.int32)
            cv2.drawContours(on_frame, [triangle_pts], 0, self.color, -1)
            # Eye (small circle)
            eye_offset_x = self.size // 5
            eye_y_offset = self.size // 12
            eye_y = body_center[1] - eye_y_offset
            eye_x = body_center[0] + eye_offset_x if self.dx > 0 else body_center[0] - eye_offset_x
            eye_radius_white = max(1, self.size // 15)
            eye_radius_pupil = max(1, self.size // 20)
            cv2.circle(on_frame, (int(eye_x), int(eye_y)), eye_radius_white, (255,255,255), -1)
            cv2.circle(on_frame, (int(eye_x), int(eye_y)), eye_radius_pupil, (0,0,0), -1)

# --- Explosion Effect ---
class ExplodingFish:
    def __init__(self, x, y, color, frame_count=12):
        self.x = x
        self.y = y
        self.color = color
        self.frame_count = frame_count
        self.current_frame = 0
        self.max_radius = 60
        self.num_particles = 12
        self.particles = [
            {
                'angle': 2 * np.pi * i / self.num_particles,
                'radius': 0
            } for i in range(self.num_particles)
        ]

    def update(self):
        self.current_frame += 1
        for p in self.particles:
            p['radius'] += self.max_radius / self.frame_count

    def draw(self, frame):
        for p in self.particles:
            px = int(self.x + np.cos(p['angle']) * p['radius'])
            py = int(self.y + np.sin(p['angle']) * p['radius'])
            cv2.circle(frame, (px, py), 8, self.color, -1)
            cv2.circle(frame, (px, py), 12, (255,255,255), 2)

    def is_done(self):
        return self.current_frame >= self.frame_count

# --- Shark Animation ---
SHARK_COLOR = (80, 80, 80)  # Gray
SHARK_BLUE_COLOR = (255, 100, 0)  # Blue (BGR)
SHARK_SIZE_RANGE = (60, 100)
SHARK_SPEED_RANGE = (2, 4)
NUM_SHARKS = 2
NUM_BLUE_SHARKS = 2
shark_list = []
blue_shark_list = []

class Shark:
    """Represents a single animated shark."""
    def __init__(self, frame_width, frame_height, color=SHARK_COLOR):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.x = random.randint(0, frame_width)
        self.y = random.randint(0, frame_height)
        self.size = random.randint(*SHARK_SIZE_RANGE)
        self.color = color
        self.dx = random.choice([-1, 1]) * random.uniform(*SHARK_SPEED_RANGE)
        self.dy = random.choice([-1, 1]) * random.uniform(1, 2)

    def update(self):
        self.x += self.dx
        self.y += self.dy
        # Boundary checks
        if self.x - self.size // 2 < 0:
            self.x = self.size // 2
            self.dx *= -1
        elif self.x + self.size // 2 > self.frame_width:
            self.x = self.frame_width - self.size // 2
            self.dx *= -1
        if self.y - self.size // 4 < 0:
            self.y = self.size // 4
            self.dy *= -1
        elif self.y + self.size // 4 > self.frame_height:
            self.y = self.frame_height - self.size // 4
            self.dy *= -1

    def draw(self, on_frame):
        body_center = (int(self.x), int(self.y))
        body_axes = (self.size // 2, self.size // 5)
        # Draw shark body (ellipse)
        cv2.ellipse(on_frame, body_center, body_axes, 0, 0, 360, self.color, -1)
        # Draw shark fin (triangle)
        fin_height = self.size // 3
        fin_base = self.size // 6
        fin_top = (body_center[0], body_center[1] - body_axes[1] - fin_height)
        fin_left = (body_center[0] - fin_base, body_center[1] - body_axes[1])
        fin_right = (body_center[0] + fin_base, body_center[1] - body_axes[1])
        fin_pts = np.array([fin_top, fin_left, fin_right], dtype=np.int32)
        cv2.drawContours(on_frame, [fin_pts], 0, (100, 100, 100), -1)
        # Draw tail (triangle, like fish but bigger)
        tail_size = self.size // 2
        if self.dx > 0:
            pt1 = (body_center[0] - body_axes[0], body_center[1])
            pt2 = (pt1[0] - tail_size, pt1[1] - tail_size // 2)
            pt3 = (pt1[0] - tail_size, pt1[1] + tail_size // 2)
        else:
            pt1 = (body_center[0] + body_axes[0], body_center[1])
            pt2 = (pt1[0] + tail_size, pt1[1] - tail_size // 2)
            pt3 = (pt1[0] + tail_size, pt1[1] + tail_size // 2)
        tail_pts = np.array([pt1, pt2, pt3], dtype=np.int32)
        cv2.drawContours(on_frame, [tail_pts], 0, self.color, -1)
        # Draw eye (white + black)
        eye_offset_x = self.size // 4
        eye_y_offset = self.size // 10
        eye_y = body_center[1] - eye_y_offset
        eye_x = body_center[0] + eye_offset_x if self.dx > 0 else body_center[0] - eye_offset_x
        cv2.circle(on_frame, (int(eye_x), int(eye_y)), max(2, self.size // 18), (255,255,255), -1)
        cv2.circle(on_frame, (int(eye_x), int(eye_y)), max(1, self.size // 30), (0,0,0), -1)

# --- Helper Functions ---
def draw_ui_buttons(frame_to_draw_on):
    """Draws static UI buttons on the frame."""
    global current_mode
    # Pen Button
    cv2.rectangle(frame_to_draw_on, (BUTTON_PEN_RECT[0], BUTTON_PEN_RECT[1]),
                  (BUTTON_PEN_RECT[0] + BUTTON_PEN_RECT[2], BUTTON_PEN_RECT[1] + BUTTON_PEN_RECT[3]),
                  PEN_BUTTON_COLOR, -1)
    cv2.putText(frame_to_draw_on, "PEN", (BUTTON_PEN_RECT[0] + 35, BUTTON_PEN_RECT[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BUTTON_TEXT_COLOR_DARK, 2)
    if current_mode == "pen":
        cv2.rectangle(frame_to_draw_on, (BUTTON_PEN_RECT[0], BUTTON_PEN_RECT[1]),
                      (BUTTON_PEN_RECT[0] + BUTTON_PEN_RECT[2], BUTTON_PEN_RECT[1] + BUTTON_PEN_RECT[3]),
                      HIGHLIGHT_COLOR, 3)
    # Eraser Button
    cv2.rectangle(frame_to_draw_on, (BUTTON_ERASER_RECT[0], BUTTON_ERASER_RECT[1]),
                  (BUTTON_ERASER_RECT[0] + BUTTON_ERASER_RECT[2], BUTTON_ERASER_RECT[1] + BUTTON_ERASER_RECT[3]),
                  ERASER_BUTTON_COLOR, -1)
    cv2.putText(frame_to_draw_on, "ERASER", (BUTTON_ERASER_RECT[0] + 20, BUTTON_ERASER_RECT[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BUTTON_TEXT_COLOR_DARK, 2)
    if current_mode == "eraser":
        cv2.rectangle(frame_to_draw_on, (BUTTON_ERASER_RECT[0], BUTTON_ERASER_RECT[1]),
                      (BUTTON_ERASER_RECT[0] + BUTTON_ERASER_RECT[2], BUTTON_ERASER_RECT[1] + BUTTON_ERASER_RECT[3]),
                      HIGHLIGHT_COLOR, 3)
    # Clear Button
    cv2.rectangle(frame_to_draw_on, (BUTTON_CLEAR_RECT[0], BUTTON_CLEAR_RECT[1]),
                  (BUTTON_CLEAR_RECT[0] + BUTTON_CLEAR_RECT[2], BUTTON_CLEAR_RECT[1] + BUTTON_CLEAR_RECT[3]),
                  CLEAR_BUTTON_COLOR, -1)
    cv2.putText(frame_to_draw_on, "CLEAR", (BUTTON_CLEAR_RECT[0] + 25, BUTTON_CLEAR_RECT[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, BUTTON_TEXT_COLOR_LIGHT, 2)

def handle_mouse_click(event, x, y, flags, param):
    """Handles mouse clicks for button interactions."""
    global current_mode, canvas, last_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if BUTTON_PEN_RECT[0] <= x <= BUTTON_PEN_RECT[0] + BUTTON_PEN_RECT[2] and \
           BUTTON_PEN_RECT[1] <= y <= BUTTON_PEN_RECT[1] + BUTTON_PEN_RECT[3]:
            current_mode = "pen"
            last_points["pen_green"] = None
            print("Mode: Pen (using GREEN object)")
        elif BUTTON_ERASER_RECT[0] <= x <= BUTTON_ERASER_RECT[0] + BUTTON_ERASER_RECT[2] and \
             BUTTON_ERASER_RECT[1] <= y <= BUTTON_ERASER_RECT[1] + BUTTON_ERASER_RECT[3]:
            current_mode = "eraser"
            print("Mode: Eraser (using WHITE object)")
        elif BUTTON_CLEAR_RECT[0] <= x <= BUTTON_CLEAR_RECT[0] + BUTTON_CLEAR_RECT[2] and \
             BUTTON_CLEAR_RECT[1] <= y <= BUTTON_CLEAR_RECT[1] + BUTTON_CLEAR_RECT[3]:
            if canvas is not None:
                canvas.fill(0) # Fill canvas with black (clears drawing)
            last_points["pen_green"] = None
            print("Canvas Cleared")

def find_color_contour(hsv_frame, lower_bound, upper_bound, min_radius_detection=10):
    """Detects color, returns center (x,y) and radius of the largest contour."""
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        if radius > min_radius_detection:
            return (int(x), int(y)), int(radius)
    return None, None

def overlay_png(bg, fg, x, y, size=None):
    """Overlay fg (with alpha) onto bg at position (x, y) (centered). Optionally resize to size (w, h)."""
    if fg is None:
        return
    fg_img = fg.copy()
    if size is not None:
        fg_img = cv2.resize(fg_img, size, interpolation=cv2.INTER_AREA)
    fh, fw = fg_img.shape[:2]
    bh, bw = bg.shape[:2]
    # Calculate top-left corner
    x1 = max(x - fw // 2, 0)
    y1 = max(y - fh // 2, 0)
    x2 = min(x1 + fw, bw)
    y2 = min(y1 + fh, bh)
    fg_x1 = 0 if x1 == x - fw // 2 else fw - (x2 - x1)
    fg_y1 = 0 if y1 == y - fh // 2 else fh - (y2 - y1)
    fg_cropped = fg_img[fg_y1:fg_y1 + (y2 - y1), fg_x1:fg_x1 + (x2 - x1)]
    bg_crop = bg[y1:y2, x1:x2]
    if fg_cropped.shape[2] == 4:
        alpha = fg_cropped[..., 3:] / 255.0
        bg_crop[:] = (1 - alpha) * bg_crop + alpha * fg_cropped[..., :3]
    else:
        bg_crop[:] = fg_cropped[..., :3]

# --- Main Application Logic ---
def run_air_painter():
    global canvas, current_mode, last_points, fish_list

    highscore = 0

    # Try index 0 with DirectShow backend (best for Windows)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Try index 1
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Try to set camera to high resolution (Full HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read initial frame.")
        cap.release()
        return
    
    frame_height, frame_width = frame.shape[:2]
    canvas = np.zeros_like(frame, dtype=np.uint8) # User's drawing canvas (black is transparent)

    # Initialize fish
    fish_list = [Fish(frame_width, frame_height) for _ in range(NUM_FISH)]
    exploding_fish_list = []  # Track active explosions
    soft_splash_list = []    # Track soft splash effects
    # Initialize sharks
    global shark_list
    shark_list = [Shark(frame_width, frame_height, SHARK_COLOR) for _ in range(NUM_SHARKS)]
    global blue_shark_list
    blue_shark_list = [Shark(frame_width, frame_height, SHARK_BLUE_COLOR) for _ in range(NUM_BLUE_SHARKS)]

    cv2.namedWindow("Air Painter", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Air Painter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.setMouseCallback("Air Painter", handle_mouse_click)  # Hide menu: disable mouse callback

    print("--- Air Painter with Fish Initialized ---")
    print("Instructions:")
    print(" - Point your finger at the camera to use as a pointer.")
    print(" - When your fingertip touches a fish, it will explode!")
    print(" - Show a WHITE object to the camera to use as an ERASER.")
    print(" - Click the on-screen buttons or use keyboard shortcuts:")
    print("   'c' or 'C' for Clear Canvas")
    print("   'q' or 'Q' to Quit")
    print("-------------------------------")
    print("If color detection is not working well, you may need to adjust the HSV values in the code.")

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    if 'mp_pose_instance' not in globals():
        global mp_pose_instance
        mp_pose_instance = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    last_ankle_pos = {'left': None, 'right': None}
    kick_cooldown = {'left': 0, 'right': 0}
    KICK_THRESHOLD = 60  # Minimum upward movement in pixels to count as a kick
    KICK_COOLDOWN_FRAMES = 15
    kick_flash = []  # List of (x, y, frame_count)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Frame not received. Exiting.")
            break

        frame = cv2.flip(frame, 1) # Mirror frame
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- MediaPipe Pose for Leg Kick Detection ---
        pose_results = mp_pose_instance.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]
        if pose_results.pose_landmarks:
            # Left ankle = 27, Right ankle = 28
            left_ankle = pose_results.pose_landmarks.landmark[27]
            right_ankle = pose_results.pose_landmarks.landmark[28]
            lx, ly = int(left_ankle.x * w), int(left_ankle.y * h)
            rx, ry = int(right_ankle.x * w), int(right_ankle.y * h)
            # Draw circles at ankles
            cv2.circle(frame, (lx, ly), 30, (0,255,0), 3)
            cv2.circle(frame, (rx, ry), 30, (255,0,0), 3)
            # Detect upward kick (y decreases by threshold)
            for side, (x, y) in zip(['left', 'right'], [(lx, ly), (rx, ry)]):
                if last_ankle_pos[side] is not None and kick_cooldown[side] == 0:
                    prev_y = last_ankle_pos[side][1]
                    if prev_y - y > KICK_THRESHOLD:
                        print(f"{side.capitalize()} leg kick detected!")
                        kick_flash.append({'x': x, 'y': y, 'frames': 0, 'color': (0,255,0) if side=='left' else (255,0,0)})
                        kick_cooldown[side] = KICK_COOLDOWN_FRAMES
                last_ankle_pos[side] = (x, y)
                if kick_cooldown[side] > 0:
                    kick_cooldown[side] -= 1
        # Draw kick flash effect
        for flash in kick_flash:
            cv2.circle(frame, (flash['x'], flash['y']), 60, flash['color'], -1)
            flash['frames'] += 1
        kick_flash = [f for f in kick_flash if f['frames'] < 8]

        # --- Create background with fish ---
        background_with_fish = frame.copy() # Fish will be drawn on top of camera feed

        # Draw and update fish
        for fish_obj in fish_list:
            fish_obj.update()
            fish_obj.draw(background_with_fish)
        # Draw and update sharks
        for shark_obj in shark_list:
            shark_obj.update()
            shark_obj.draw(background_with_fish)
        for blue_shark_obj in blue_shark_list:
            blue_shark_obj.update()
            blue_shark_obj.draw(background_with_fish)

        # Draw and update explosions
        for explosion in exploding_fish_list:
            explosion.update()
            explosion.draw(background_with_fish)
        # Remove finished explosions
        exploding_fish_list = [e for e in exploding_fish_list if not e.is_done()]
        # Draw and update soft splashes
        for splash in soft_splash_list:
            splash.update()
            splash.draw(background_with_fish)
        soft_splash_list = [s for s in soft_splash_list if not s.is_done()]

        # --- MULTI-POINTER PALM DETECTION ---
        # Use MediaPipe Hands for robust palm detection
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        if 'mp_hands_instance' not in globals():
            global mp_hands_instance
            mp_hands_instance = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
        palm_centers = []
        results = mp_hands_instance.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Palm center: average of landmarks 0, 5, 9, 13, 17
                h, w = frame.shape[:2]
                palm_points = [hand_landmarks.landmark[j] for j in [0, 5, 9, 13, 17]]
                cx = int(np.mean([p.x for p in palm_points]) * w)
                cy = int(np.mean([p.y for p in palm_points]) * h)
                palm_centers.append((cx, cy))
                # Draw pointer: triangle with nice opacity outline
                tri_height = 120
                tri_base = 90
                pt_top = (cx, cy - tri_height // 2)
                pt_left = (cx - tri_base // 2, cy + tri_height // 2)
                pt_right = (cx + tri_base // 2, cy + tri_height // 2)
                triangle = np.array([[pt_top, pt_left, pt_right]], dtype=np.int32)
                overlay = background_with_fish.copy()
                if i == 0:
                    fill_color = (0, 255, 255)
                    outline_color = (0, 255, 255)
                else:
                    fill_color = (255, 0, 255)
                    outline_color = (255, 0, 255)
                cv2.fillPoly(overlay, triangle, fill_color)
                cv2.addWeighted(overlay, 0.25, background_with_fish, 0.75, 0, background_with_fish)
                cv2.polylines(background_with_fish, triangle, isClosed=True, color=outline_color, thickness=6, lineType=cv2.LINE_AA)

        # --- PALM-FISH COLLISION DETECTION ---
        COLLISION_RADIUS = 120  # Sensitivity: treat pointer as a circle of this radius
        COLLISION_SENSITIVITY = 1.5  # Further inflate axes for more sensitivity
        for palm_center in palm_centers:
            px, py = palm_center
            hit = False
            for idx, fish_obj in enumerate(fish_list):
                cx, cy = int(fish_obj.x), int(fish_obj.y)
                a = fish_obj.size // 2
                b = fish_obj.size // 4
                if a > 0 and b > 0:
                    # Check if the ellipse and pointer circle overlap (approximate by inflating axes)
                    if ((px - cx) ** 2) / ((COLLISION_SENSITIVITY * (a + COLLISION_RADIUS)) ** 2) + ((py - cy) ** 2) / ((COLLISION_SENSITIVITY * (b + COLLISION_RADIUS)) ** 2) <= 1:
                        # Trigger explosion
                        exploding_fish_list.append(ExplodingFish(cx, cy, fish_obj.color))
                        # Respawn fish at a new location
                        fish_list[idx] = Fish(frame_width, frame_height)
                        highscore += 1
                        hit = True
                        break
            if not hit:
                # If no fish was hit, show a soft splash at the pointer
                soft_splash_list.append(SoftSplash(px, py))

        # --- Eraser (white object) is now disabled ---
        # center, radius = find_color_contour(hsv_frame, WHITE_LOWER, WHITE_UPPER, min_radius_detection=15)
        # if center:
        #     cv2.circle(canvas, center, ERASER_THICKNESS // 2, ERASER_COLOR_WHITE_BGR, -1) # Erase by drawing white

        # --- Display Logic ---
        # Hide UI buttons: do not draw them
        # draw_ui_buttons(background_with_fish)

        # Combine the background_with_fish (camera + fish + UI) with the user's drawing_canvas
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, drawing_mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
        drawing_mask_inv = cv2.bitwise_not(drawing_mask)
        final_background_part = cv2.bitwise_and(background_with_fish, background_with_fish, mask=drawing_mask_inv)
        final_drawing_part = cv2.bitwise_and(canvas, canvas, mask=drawing_mask)
        display_frame = cv2.add(final_background_part, final_drawing_part)

        # Draw highscore at top right
        score_text = f"Score: {highscore}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        text_size, _ = cv2.getTextSize(score_text, font, font_scale, thickness)
        text_x = (background_with_fish.shape[1] - text_size[0]) // 2
        text_y = 50
        cv2.putText(background_with_fish, score_text, (text_x, text_y), font, font_scale, (0,0,0), thickness+2, cv2.LINE_AA)
        cv2.putText(background_with_fish, score_text, (text_x, text_y), font, font_scale, (255,255,0), thickness, cv2.LINE_AA)

        cv2.imshow("Air Painter", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            if canvas is not None: canvas.fill(0)
            last_points["pen_green"] = None; print("Canvas Cleared")

    cap.release()
    cv2.destroyAllWindows()
    print("Air Painter closed.")

class SoftSplash:
    def __init__(self, x, y, color=(200, 200, 255), frame_count=10):
        self.x = x
        self.y = y
        self.color = color
        self.frame_count = frame_count
        self.current_frame = 0
        self.max_radius = 80
    def update(self):
        self.current_frame += 1
    def draw(self, frame):
        alpha = max(0, 1 - self.current_frame / self.frame_count)
        overlay = frame.copy()
        radius = int(self.max_radius * (self.current_frame / self.frame_count))
        cv2.circle(overlay, (self.x, self.y), radius, self.color, -1)
        cv2.addWeighted(overlay, 0.2 * alpha, frame, 1 - 0.2 * alpha, 0, frame)
    def is_done(self):
        return self.current_frame >= self.frame_count

if __name__ == "__main__":
    run_air_painter()
