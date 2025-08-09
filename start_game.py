"""
Script to start the game and AI.
This is a convenience script that handles the startup process.
"""

import os
import time
import argparse
import subprocess
from pathlib import Path

def kill_processes():
    """Kill existing scrcpy and adb processes."""
    print("Killing existing processes...")
    os.system('taskkill /IM scrcpy.exe /F')
    os.system('taskkill /IM adb.exe /F')

def start_scrcpy():
    """Start scrcpy for screen mirroring."""
    print("Starting scrcpy...")
    process = subprocess.Popen(["scrcpy", "--max-size", "960"])
    return process

def start_ai(duration=None):
    """Start the AI."""
    print("Starting AI...")
    cmd = ["python", "main.py", "--mode", "play"]
    if duration:
        cmd.extend(["--duration", str(duration)])
    
    process = subprocess.Popen(cmd)
    return process

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start the game and AI")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration in seconds to run the AI")
    parser.add_argument("--no-scrcpy", action="store_true",
                        help="Don't start scrcpy (use if it's already running)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("weights").mkdir(exist_ok=True)
    
    # Kill existing processes
    if not args.no_scrcpy:
        kill_processes()
    
    # Start scrcpy
    scrcpy_process = None
    if not args.no_scrcpy:
        scrcpy_process = start_scrcpy()
        # Wait for scrcpy to start
        time.sleep(5)
    
    # Start AI
    ai_process = start_ai(args.duration)
    
    try:
        # Wait for AI to finish
        ai_process.wait()
    except KeyboardInterrupt:
        print("Interrupted by user")
        if ai_process:
            ai_process.terminate()
    finally:
        # Clean up
        if scrcpy_process:
            scrcpy_process.terminate()
        
        print("Done")

if __name__ == "__main__":
    main()