#!/bin/bash

# Cross-Platform Hand Gesture Control Setup Script
# Compatible with macOS, Linux, and Windows (via Git Bash)

echo "=== Cross-Platform Hand Gesture Control Setup ==="
echo "Detected OS: $(uname -s)"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv gesture_env

# Activate virtual environment
echo "Activating virtual environment..."
case "$(uname -s)" in
    Darwin|Linux)
        source gesture_env/bin/activate
        ;;
    MINGW*|CYGWIN*|MSYS*)
        source gesture_env/Scripts/activate
        ;;
esac

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt

# Platform-specific setup
case "$(uname -s)" in
    Darwin)
        echo "macOS detected - No additional setup needed"
        echo "Note: You may need to grant camera permissions in System Preferences"
        ;;
    Linux)
        echo "Linux detected - Installing additional dependencies..."
        # Check if running on Ubuntu/Debian
        if command -v apt-get &> /dev/null; then
            echo "Installing alsa-utils for audio control..."
            sudo apt-get update
            sudo apt-get install -y alsa-utils
        # Check if running on Fedora/CentOS
        elif command -v dnf &> /dev/null; then
            echo "Installing alsa-utils for audio control..."
            sudo dnf install -y alsa-utils
        elif command -v yum &> /dev/null; then
            echo "Installing alsa-utils for audio control..."
            sudo yum install -y alsa-utils
        else
            echo "Please install alsa-utils manually for audio control"
        fi
        ;;
    MINGW*|CYGWIN*|MSYS*)
        echo "Windows detected - No additional setup needed"
        ;;
esac

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To run the application:"
echo "1. Activate virtual environment:"
case "$(uname -s)" in
    Darwin|Linux)
        echo "   source gesture_env/bin/activate"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        echo "   source gesture_env/Scripts/activate"
        ;;
esac
echo "2. Run the program:"
echo "   python3 main.py"
echo ""
echo "Controls:"
echo "- Open all fingers on right hand: Open browser"
echo "- All fingers open: Move cursor"
echo "- Only index finger: Click"
echo "- Two hands distance: Control volume"
echo "- Make fist for 5 seconds: Exit"
echo "- Press 'q' to quit"
echo ""
echo "Note: Make sure your camera is working and permissions are granted."