import cv2
import numpy as np

# --- Global Variables ---
# Drawing modes
current_mode = "pen"  # "pen" or "eraser"

# Colors (BGR format for OpenCV drawing)
PEN_COLOR_GREEN_BGR = (0, 255, 0)    # Green for pen
ERASER_COLOR_WHITE_BGR = (255, 255, 255) # White for eraser (draws white)
BUTTON_TEXT_COLOR_DARK = (0,0,0)
BUTTON_TEXT_COLOR_LIGHT = (255,255,255)
HIGHLIGHT_COLOR = (255, 255, 0) # Cyan for highlighting active button or detected object
CLEAR_BUTTON_COLOR = (50, 50, 200) # Reddish for clear button
PEN_BUTTON_COLOR = (0, 200, 0) # Green for pen button
ERASER_BUTTON_COLOR = (200, 200, 200) # Light gray for eraser button


# Thickness
PEN_THICKNESS = 5
ERASER_THICKNESS = 50 # Radius of the eraser circle, so effective thickness is its diameter

# HSV Color Ranges for Detection
# IMPORTANT: These values might need tuning based on your specific colored objects and lighting.
# You can use an online HSV color picker to find suitable ranges for your objects.

# Green (for pen)
# Hue: 35-85 (covers a good range of greens)
# Saturation: 90-255 (moderately to highly saturated greens)
# Value: 90-255 (moderately bright to very bright greens)
GREEN_LOWER = np.array([35, 90, 90])
GREEN_UPPER = np.array([85, 255, 255])

# White (for eraser)
# Hue: 0-180 (any hue, as white is achromatic)
# Saturation: 0-80 (low saturation, as white is not colorful)
# Value: 160-255 (high brightness, as white is bright)
# Note: Detecting "white" reliably can be tricky due to lighting variations.
# A dedicated, non-reflective white object works best.
WHITE_LOWER = np.array([0, 0, 160])
WHITE_UPPER = np.array([180, 80, 255])

# Canvas for drawing
canvas = None
# Stores the last point of the current stroke for continuous drawing
# Key: 'pen_green'. Value: (x,y) or None if stroke is broken
last_points = {
    "pen_green": None,
    # Eraser currently "dabs" (draws circles), so doesn't need last_point for line drawing
}

# Button Definitions (x_start, y_start, width, height)
BUTTON_MARGIN = 20
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 50
BUTTON_PEN_RECT = (BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT)
BUTTON_ERASER_RECT = (BUTTON_MARGIN, BUTTON_MARGIN * 2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT)
BUTTON_CLEAR_RECT = (BUTTON_MARGIN, BUTTON_MARGIN * 3 + BUTTON_HEIGHT * 2, BUTTON_WIDTH, BUTTON_HEIGHT)

# --- Helper Functions ---

def draw_ui_buttons(frame_to_draw_on):
    """Draws static UI buttons on the frame."""
    global current_mode

    # Pen Button
    pen_text_color = BUTTON_TEXT_COLOR_DARK
    cv2.rectangle(frame_to_draw_on, (BUTTON_PEN_RECT[0], BUTTON_PEN_RECT[1]),
                  (BUTTON_PEN_RECT[0] + BUTTON_PEN_RECT[2], BUTTON_PEN_RECT[1] + BUTTON_PEN_RECT[3]),
                  PEN_BUTTON_COLOR, -1)
    cv2.putText(frame_to_draw_on, "PEN",
                (BUTTON_PEN_RECT[0] + 35, BUTTON_PEN_RECT[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, pen_text_color, 2)
    if current_mode == "pen":
        cv2.rectangle(frame_to_draw_on, (BUTTON_PEN_RECT[0], BUTTON_PEN_RECT[1]),
                      (BUTTON_PEN_RECT[0] + BUTTON_PEN_RECT[2], BUTTON_PEN_RECT[1] + BUTTON_PEN_RECT[3]),
                      HIGHLIGHT_COLOR, 3) # Highlight border if active

    # Eraser Button
    eraser_text_color = BUTTON_TEXT_COLOR_DARK
    cv2.rectangle(frame_to_draw_on, (BUTTON_ERASER_RECT[0], BUTTON_ERASER_RECT[1]),
                  (BUTTON_ERASER_RECT[0] + BUTTON_ERASER_RECT[2], BUTTON_ERASER_RECT[1] + BUTTON_ERASER_RECT[3]),
                  ERASER_BUTTON_COLOR, -1)
    cv2.putText(frame_to_draw_on, "ERASER",
                (BUTTON_ERASER_RECT[0] + 20, BUTTON_ERASER_RECT[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, eraser_text_color, 2)
    if current_mode == "eraser":
        cv2.rectangle(frame_to_draw_on, (BUTTON_ERASER_RECT[0], BUTTON_ERASER_RECT[1]),
                      (BUTTON_ERASER_RECT[0] + BUTTON_ERASER_RECT[2], BUTTON_ERASER_RECT[1] + BUTTON_ERASER_RECT[3]),
                      HIGHLIGHT_COLOR, 3) # Highlight border if active

    # Clear Button
    clear_text_color = BUTTON_TEXT_COLOR_LIGHT
    cv2.rectangle(frame_to_draw_on, (BUTTON_CLEAR_RECT[0], BUTTON_CLEAR_RECT[1]),
                  (BUTTON_CLEAR_RECT[0] + BUTTON_CLEAR_RECT[2], BUTTON_CLEAR_RECT[1] + BUTTON_CLEAR_RECT[3]),
                  CLEAR_BUTTON_COLOR, -1)
    cv2.putText(frame_to_draw_on, "CLEAR",
                (BUTTON_CLEAR_RECT[0] + 25, BUTTON_CLEAR_RECT[1] + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, clear_text_color, 2)


def handle_mouse_click(event, x, y, flags, param):
    """Handles mouse clicks for button interactions."""
    global current_mode, canvas, last_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check Pen button
        if BUTTON_PEN_RECT[0] <= x <= BUTTON_PEN_RECT[0] + BUTTON_PEN_RECT[2] and \
           BUTTON_PEN_RECT[1] <= y <= BUTTON_PEN_RECT[1] + BUTTON_PEN_RECT[3]:
            current_mode = "pen"
            last_points["pen_green"] = None # Reset last point on mode switch
            print("Mode: Pen (using GREEN object)")

        # Check Eraser button
        elif BUTTON_ERASER_RECT[0] <= x <= BUTTON_ERASER_RECT[0] + BUTTON_ERASER_RECT[2] and \
             BUTTON_ERASER_RECT[1] <= y <= BUTTON_ERASER_RECT[1] + BUTTON_ERASER_RECT[3]:
            current_mode = "eraser"
            # No last_point needed for eraser as it "dabs"
            print("Mode: Eraser (using WHITE object)")

        # Check Clear button
        elif BUTTON_CLEAR_RECT[0] <= x <= BUTTON_CLEAR_RECT[0] + BUTTON_CLEAR_RECT[2] and \
             BUTTON_CLEAR_RECT[1] <= y <= BUTTON_CLEAR_RECT[1] + BUTTON_CLEAR_RECT[3]:
            if canvas is not None:
                canvas.fill(0) # Fill canvas with black (clears drawing)
            last_points["pen_green"] = None # Reset pen stroke
            print("Canvas Cleared")


def find_color_contour(hsv_frame, lower_bound, upper_bound, min_radius_detection=10):
    """
    Detects a color within the given HSV range in the hsv_frame.
    Returns the center (x,y) and radius of the largest detected contour if it meets min_radius_detection.
    Otherwise, returns None, None.
    """
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    # Morphological operations to reduce noise and improve contour detection
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the minimum enclosing circle for the largest contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        if radius > min_radius_detection:
            return (int(x), int(y)), int(radius)
    return None, None

# --- Main Application Logic ---
def run_air_painter():
    global canvas, current_mode, last_points

    # Initialize webcam
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    if not cap.isOpened():
        print("Error: Cannot open webcam. Please check if it's connected and not used by another application.")
        return

    # Read the first frame to get dimensions for the canvas
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read initial frame from webcam.")
        cap.release()
        return

    # Initialize canvas as a black image with the same dimensions as the webcam frame
    canvas = np.zeros_like(frame, dtype=np.uint8)

    # Create window and set mouse callback for button clicks
    cv2.namedWindow("Air Painter")
    cv2.setMouseCallback("Air Painter", handle_mouse_click)

    print("--- Air Painter Initialized ---")
    print("Instructions:")
    print(" - Show a GREEN object to the camera to use as a PEN.")
    print(" - Show a WHITE object to the camera to use as an ERASER.")
    print(" - Click the on-screen buttons or use keyboard shortcuts:")
    print("   'p' or 'P' for Pen Mode")
    print("   'e' or 'E' for Eraser Mode")
    print("   'c' or 'C' for Clear Canvas")
    print("   'q' or 'Q' to Quit")
    print("-------------------------------")
    print("If color detection is not working well, you may need to adjust the HSV values in the code.")


    while True:
        ret, frame = cap.read() # Read a new frame from the webcam
        if not ret:
            print("Error: Frame not received from webcam. Exiting loop.")
            break

        frame = cv2.flip(frame, 1) # Mirror the frame for a more intuitive drawing experience
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert frame to HSV color space for detection

        detected_tool_center = None
        detected_tool_radius = None # For visual feedback on the frame

        if current_mode == "pen":
            center, radius = find_color_contour(hsv_frame, GREEN_LOWER, GREEN_UPPER)
            if center:
                detected_tool_center = center
                detected_tool_radius = radius
                # For continuous drawing: if there was a last point, draw a line to the current point
                if last_points["pen_green"] is not None:
                    cv2.line(canvas, last_points["pen_green"], center, PEN_COLOR_GREEN_BGR, PEN_THICKNESS)
                else: # If it's the start of a new stroke, draw a small circle to mark the beginning
                     cv2.circle(canvas, center, PEN_THICKNESS // 2, PEN_COLOR_GREEN_BGR, -1)
                last_points["pen_green"] = center # Update the last point
            else:
                last_points["pen_green"] = None # Pen is lifted (no green detected)

        elif current_mode == "eraser":
            # Eraser might need a slightly larger object or more lenient detection
            center, radius = find_color_contour(hsv_frame, WHITE_LOWER, WHITE_UPPER, min_radius_detection=15)
            if center:
                detected_tool_center = center
                detected_tool_radius = radius
                # Erase by drawing white circles on the canvas
                cv2.circle(canvas, center, ERASER_THICKNESS // 2, ERASER_COLOR_WHITE_BGR, -1)
            # No need to store last_point for eraser if it's just dabbing circles

        # --- Display Logic ---

        # Visual feedback for detected tool on the live camera frame (not the canvas)
        if detected_tool_center and detected_tool_radius:
            feedback_color = PEN_COLOR_GREEN_BGR if current_mode == "pen" else ERASER_COLOR_WHITE_BGR
            cv2.circle(frame, detected_tool_center, detected_tool_radius, HIGHLIGHT_COLOR, 2) # Outline detected object
            cv2.circle(frame, detected_tool_center, 5, feedback_color, -1) # Small dot at center

        # Draw UI buttons on the frame
        draw_ui_buttons(frame)

        # Combine the camera feed with the drawing canvas
        # Create a mask where the canvas has drawings (is not black)
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, drawing_mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY) # Pixels > 1 are part of drawing
        drawing_mask_inv = cv2.bitwise_not(drawing_mask) # Inverse mask for areas without drawing

        # Black-out the area of the drawing on the live frame
        frame_masked_background = cv2.bitwise_and(frame, frame, mask=drawing_mask_inv)
        # Add the drawing from the canvas onto this blacked-out area
        display_frame = cv2.add(frame_masked_background, canvas)

        cv2.imshow("Air Painter", display_frame)
        # For debugging, you can uncomment these lines to see individual components:
        # cv2.imshow("Canvas Only", canvas)
        # if current_mode == "pen":
        #     cv2.imshow("Green Mask", cv2.inRange(hsv_frame, GREEN_LOWER, GREEN_UPPER))
        # elif current_mode == "eraser":
        #     cv2.imshow("White Mask", cv2.inRange(hsv_frame, WHITE_LOWER, WHITE_UPPER))

        # Handle keyboard input
        key = cv2.waitKey(30) & 0xFF # Wait for 30ms, capture key press
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('p') or key == ord('P'):
            current_mode = "pen"
            last_points["pen_green"] = None
            print("Mode: Pen (using GREEN object)")
        elif key == ord('e') or key == ord('E'):
            current_mode = "eraser"
            print("Mode: Eraser (using WHITE object)")
        elif key == ord('c') or key == ord('C'):
            if canvas is not None:
                canvas.fill(0) # Clear canvas
            last_points["pen_green"] = None
            print("Canvas Cleared")

    # Cleanup
    cap.release() # Release the webcam
    cv2.destroyAllWindows() # Close all OpenCV windows
    print("Air Painter closed.")

if __name__ == "__main__":
    run_air_painter()
