Sure! Here's the English version of your documentation:

---

# Modern Hand Gesture to Watch YouTube

Computer Vision Project to Control YouTube Using Hand Gestures

## Getting Started

This guide will help you set up and run the Hand Gesture Control program on various operating systems (macOS, Linux, Windows).

### Prerequisites

* Python 3.7 or higher
* pip (Python package manager)
* A working camera with proper permission granted

### Automatic Setup (Recommended)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/bagaspra16/vision-python.git
   cd vision-python
   ```

2. **Run the setup script:**

   ```bash
   bash setup.sh
   ```

   Optional for Linux/macOS users:

   ```bash
   chmod +x setup.sh && ./setup.sh
   ```

   This script will:

   * Create a virtual environment (`gesture_env`)
   * Activate the virtual environment
   * Install all dependencies from `Requirements.txt`
   * Perform additional OS-specific setup (e.g., install `alsa-utils` on Linux)

3. **Activate the virtual environment (if not already):**

   * **macOS/Linux:**

     ```bash
     source gesture_env/bin/activate
     ```
   * **Windows (Git Bash/PowerShell):**

     ```bash
     source gesture_env/Scripts/activate
     ```

4. **Run the program:**

   ```bash
   python3 main.py
   ```

### Manual Setup (Alternative)

1. Create and activate the virtual environment:

   ```bash
   python3 -m venv gesture_env
   source gesture_env/bin/activate  # macOS/Linux
   # or
   source gesture_env/Scripts/activate  # Windows
   ```

2. Install all dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r Requirements.txt
   ```

### Controls & Usage

Once the program is running, you can control your computer using hand gestures in front of the camera:

* **Open all fingers of the right hand:** Open browser (YouTube)
* **All fingers of right hand open:** Move the mouse cursor
* **Only the index finger extended:** Left click
* **Distance between both hands:** Control volume (farther = increase, closer = decrease)
* **Make a fist with the right hand for 5 seconds:** Exit the program
* **Move hand up/down on the screen:** Auto-scroll the page
* **Press 'q' key on keyboard:** Quit the application

**Notes:**

* Make sure the camera is granted access in your OS settings.
* On Linux, ensure `alsa-utils` is installed for volume control to work.
* On macOS, you may need to allow camera and input control access via System Preferences.

### Troubleshooting

* If the camera is not detected, ensure it's not being used by another application.
* If gestures aren't responsive, ensure proper lighting and that your hand is clearly visible to the camera.
* For library-related errors, make sure all dependencies are installed in the virtual environment.

---

This program uses OpenCV, MediaPipe, pyautogui, and other libraries to detect hand gestures and control the computer cross-platform.

---

