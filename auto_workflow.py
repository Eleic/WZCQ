"""
Automated workflow script for the WZCQ project.
This script automates the entire workflow from data collection to training.
"""

import os
import time
import argparse
import subprocess
from pathlib import Path
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auto_workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("auto_workflow")

def load_config():
    """Load configuration from file."""
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        logger.warning("Config file not found, using defaults")
        return {}

def save_config(config):
    """Save configuration to file."""
    config_path = "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def run_command(cmd, timeout=None):
    """Run a command and return its output."""
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(timeout=timeout)
        
        if process.returncode != 0:
            logger.error(f"Command failed with return code {process.returncode}")
            logger.error(f"stderr: {stderr}")
            return False, stderr
        
        return True, stdout
    except subprocess.TimeoutExpired:
        process.kill()
        logger.warning(f"Command timed out after {timeout} seconds")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return False, str(e)

def collect_data(duration):
    """Collect training data by playing the game."""
    logger.info(f"Collecting training data for {duration} seconds")
    
    # Start scrcpy
    run_command(["python", "start_game.py", "--duration", str(duration)])
    
    logger.info("Data collection completed")
    return True

def process_data():
    """Process collected training data."""
    logger.info("Processing training data")
    
    success, output = run_command(["python", "main.py", "--mode", "process"])
    
    if success:
        logger.info("Data processing completed")
        return True
    else:
        logger.error("Data processing failed")
        return False

def train_model(epochs):
    """Train the model on processed data."""
    logger.info(f"Training model for {epochs} epochs")
    
    success, output = run_command(["python", "main.py", "--mode", "train", "--epochs", str(epochs)])
    
    if success:
        logger.info("Model training completed")
        return True
    else:
        logger.error("Model training failed")
        return False

def schedule_runs(times, duration):
    """Schedule runs at specific times."""
    logger.info(f"Scheduling runs at times: {times}")
    
    for time_str in times:
        success, output = run_command([
            "python", "main.py", "--mode", "schedule",
            "--schedule", time_str, "--duration", str(duration)
        ])
        
        if success:
            logger.info(f"Scheduled run at {time_str}")
        else:
            logger.error(f"Failed to schedule run at {time_str}")
    
    return True

def automated_workflow(args):
    """Run the automated workflow."""
    logger.info("Starting automated workflow")
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("weights").mkdir(exist_ok=True)
    
    # Load configuration
    config = load_config()
    
    # Update configuration with command line arguments
    if "automation" not in config:
        config["automation"] = {}
    
    if args.data_collection_duration:
        config["automation"]["data_collection_duration"] = args.data_collection_duration
    
    if args.training_epochs:
        config["training"] = config.get("training", {})
        config["training"]["epochs"] = args.training_epochs
    
    # Save updated configuration
    save_config(config)
    
    # Run workflow steps
    if args.collect_data:
        collect_data(config["automation"].get("data_collection_duration", 3600))
    
    if args.process_data:
        process_data()
    
    if args.train_model:
        train_model(config["training"].get("epochs", 1))
    
    if args.schedule and args.schedule_times:
        schedule_runs(
            args.schedule_times,
            config["automation"].get("data_collection_duration", 3600)
        )
    
    logger.info("Automated workflow completed")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated workflow for WZCQ")
    parser.add_argument("--collect-data", action="store_true",
                        help="Collect training data")
    parser.add_argument("--process-data", action="store_true",
                        help="Process collected data")
    parser.add_argument("--train-model", action="store_true",
                        help="Train the model")
    parser.add_argument("--schedule", action="store_true",
                        help="Schedule runs")
    parser.add_argument("--data-collection-duration", type=int, default=None,
                        help="Duration in seconds for data collection")
    parser.add_argument("--training-epochs", type=int, default=None,
                        help="Number of epochs for training")
    parser.add_argument("--schedule-times", nargs="+", default=None,
                        help="Times to schedule runs (HH:MM format)")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps")
    
    args = parser.parse_args()
    
    # If --all is specified, run all steps
    if args.all:
        args.collect_data = True
        args.process_data = True
        args.train_model = True
    
    # If no steps specified, show help
    if not (args.collect_data or args.process_data or args.train_model or args.schedule):
        parser.print_help()
        return
    
    automated_workflow(args)

if __name__ == "__main__":
    main()