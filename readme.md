## Aqua Poke
# Fish & Shark Interactive Game (Webcam, OpenCV, MediaPipe)

This is a webcam-based interactive game where you use your hands (and optionally leg kicks) to interact with animated fish and sharks on the screen. The game uses real-time hand and pose detection (MediaPipe) to let you "poke" or "kick" the fish, causing them to explode and increase your score. Some fish use PNG images for variety, and there are animated sharks for extra fun!

## Features

- **Webcam-based Interaction:** Use your hands (and feet) to interact with the game.
- **Hand Detection:** Robust palm detection using MediaPipe Hands.
- **Leg Kick Detection:** Detects leg kicks using MediaPipe Pose.
- **Animated Fish & Sharks:** Fish and sharks swim around; fish can be exploded by poking or kicking.
- **Fish Variations:** Supports multiple PNG fish images and several OpenCV-drawn fish styles.
- **Score System:** Score increases each time you explode a fish. Score is displayed at the top center.
- **Visual Effects:** Explosions, splashes, and pointer triangles with nice outlines.
- **Full Screen Experience:** The game runs in full screen for immersion.

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- MediaPipe

Install with:
```bash
pip install opencv-python numpy mediapipe
```

## Setup Instructions

1. **Clone or Download the Project.**
2. **(Optional) Add PNG Fish Images:** Place `fish.png`, `fish2.png`, `fish3.png`, `fish4.png` in the project folder for more fish variety.
3. **Run the Game:**
   ```bash
   python main.py
   ```

## How to Play

- **Hand Interaction:** Show your open palm(s) to the camera. Triangles will appear at your palm(s). Move your hand to poke fish—if you touch a fish, it explodes and you score a point!
- **Leg Kick Interaction:** Kick your leg upward in view of the camera. If your ankle moves up quickly, a flash will appear and you can use this to interact with fish (if you connect the logic).
- **Score:** Your score is shown at the top center. Try to get the highest score by exploding as many fish as possible!
- **Visual Feedback:** Explosions, splashes, and pointer triangles provide real-time feedback.

## Customization

- **Add More Fish:** Add PNG images named `fish2.png`, `fish3.png`, `fish4.png` for more fish types.
- **Change Fish/Pointer Size:** Adjust the `self.size` in the `Fish` class or triangle size in the code.
- **Tune Detection:** You can adjust MediaPipe detection confidence or explosion sensitivity in the code.

## Troubleshooting

- **Webcam Not Detected:** Make sure your webcam is connected and not used by another app.
- **Hand/Leg Not Detected:** Ensure good lighting and that your hand/leg is visible to the camera.
- **Dependencies:** Make sure you installed all requirements.

## Exiting

- Press `q` to quit the game.

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:

```bash
deactivate