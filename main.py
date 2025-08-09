"""
Main module for the WZCQ project.
This module ties everything together and provides the main entry point.
"""

import os
import time
import argparse
import threading
import torch
import numpy as np
from PIL import Image

from config_manager import config
from logger import logger
from game_controller import game_controller
from data_processor import data_processor
from ai_agent import ai_agent
from scheduler import scheduler
from Batch import create_masks
from 取训练数据 import 取图

class GameSession:
    """Manages a game session."""
    
    def __init__(self):
        """Initialize a game session."""
        self.running = False
        self.session_thread = None
        self.session_dir = None
        self.record_file = None
        self.image_tensor = torch.Tensor(0)
        self.operation_sequence = np.ones((1, ))
        self.count = 0
        self.old_instruction = '移动停'
        self.auto_mode = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.window_name = config.get("device", "window_name")
        self.screenshot_interval = config.get("automation", "screenshot_interval")
        self.session_duration = config.get("automation", "session_duration")
    
    def start(self):
        """Start a game session."""
        if self.running:
            logger.warning("Game session is already running")
            return False
        
        # Create session directory
        self.session_dir = data_processor.create_session_directory()
        self.record_file = open(os.path.join(self.session_dir, "_操作数据.json"), 'w+')
        
        # Initialize tensors
        self.image_tensor = torch.Tensor(0)
        self.operation_sequence = np.ones((1, ))
        self.operation_sequence[0] = 128  # Initial operation
        self.count = 0
        self.old_instruction = '移动停'
        self.auto_mode = 0
        
        # Start keyboard listener
        game_controller.start_keyboard_listener()
        
        # Start session thread
        self.running = True
        self.session_thread = threading.Thread(target=self._session_loop)
        self.session_thread.daemon = True
        self.session_thread.start()
        
        logger.info("Game session started")
        return True
    
    def stop(self):
        """Stop the game session."""
        if not self.running:
            logger.warning("Game session is not running")
            return False
        
        # Stop session
        self.running = False
        if self.session_thread:
            self.session_thread.join(timeout=5)
        
        # Close record file
        if self.record_file:
            self.record_file.close()
            self.record_file = None
        
        # Stop keyboard listener
        game_controller.stop_keyboard_listener()
        
        logger.info("Game session stopped")
        return True
    
    def _session_loop(self):
        """Main session loop."""
        start_time = time.time()
        
        try:
            for i in range(1000000):  # Large number to run indefinitely
                # Check if session should stop
                if not self.running:
                    break
                
                # Check if session duration exceeded
                if self.session_duration and (time.time() - start_time) > self.session_duration:
                    logger.info(f"Session duration of {self.session_duration} seconds exceeded")
                    self.running = False
                    break
                
                # Check if AI is enabled
                if not game_controller.is_ai_enabled():
                    logger.info("AI disabled, stopping session")
                    self.running = False
                    break
                
                # Capture screenshot
                try:
                    img = 取图(self.window_name)
                except Exception as e:
                    logger.error(f"Failed to capture screenshot: {e}")
                    self.running = False
                    break
                
                # Start timing for this iteration
                iter_start = time.time()
                
                # Process image
                if self.image_tensor.shape[0] == 0:
                    # First image
                    img_array = np.array(img)
                    img_tensor = torch.from_numpy(img_array).to(self.device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                    features = data_processor.preprocess_image(img)
                    self.image_tensor = features
                elif self.image_tensor.shape[0] < 300:
                    # Add to tensor
                    img_array = np.array(img)
                    img_tensor = torch.from_numpy(img_array).to(self.device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                    features = data_processor.preprocess_image(img)
                    self.image_tensor = torch.cat((self.image_tensor, features), 0)
                    self.operation_sequence = np.append(self.operation_sequence, action)
                else:
                    # Maintain fixed size by removing oldest
                    img_array = np.array(img)
                    img_tensor = torch.from_numpy(img_array).to(self.device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                    features = data_processor.preprocess_image(img)
                    self.image_tensor = self.image_tensor[1:300, :]
                    self.operation_sequence = self.operation_sequence[1:300]
                    self.operation_sequence = np.append(self.operation_sequence, action)
                    self.image_tensor = torch.cat((self.image_tensor, features), 0)
                
                # Create masks
                operation_tensor = torch.from_numpy(self.operation_sequence.astype(np.int64)).to(self.device)
                src_mask, trg_mask = create_masks(operation_tensor.unsqueeze(0), operation_tensor.unsqueeze(0), self.device)
                
                # Select action
                action, action_probs, evaluation = ai_agent.select_action(
                    self.image_tensor, self.operation_sequence, trg_mask
                )
                
                # Perform periodic actions
                if self.count % 50 == 0 and self.count != 0:
                    game_controller.perform_periodic_actions()
                    game_controller.send_command('移动停')
                    logger.debug(f"Performed periodic actions, old instruction: {self.old_instruction}")
                    time.sleep(0.02)
                    game_controller.send_movement(self.old_instruction)
                
                # Process actions
                if self.count % 1 == 0:
                    # Get instruction from action
                    instruction = data_processor.数_词表[str(action)]
                    instruction_parts = instruction.split('_')
                    
                    # Create operation dictionary
                    operation_dict = {
                        "图片号": str(i),
                        "移动操作": "无移动",
                        "动作操作": "无动作"
                    }
                    
                    # Process manual input
                    movement_result = game_controller.get_movement_direction()
                    next_operation = game_controller.get_next_operation()
                    attack_active = game_controller.is_attack_active()
                    
                    if movement_result != '' or next_operation or attack_active:
                        # Manual mode
                        if movement_result == '':
                            operation_dict['移动操作'] = instruction_parts[0]
                        else:
                            operation_dict['移动操作'] = movement_result
                        
                        if next_operation:
                            operation_dict['动作操作'] = next_operation
                        elif attack_active:
                            operation_dict['动作操作'] = '攻击'
                        else:
                            operation_dict['动作操作'] = '无动作'
                        
                        # Save image
                        img_path = data_processor.save_image(img, self.session_dir, i)
                        
                        # Set end flag
                        if self.auto_mode == 0:
                            operation_dict['结束'] = 1
                        else:
                            operation_dict['结束'] = 0
                        
                        self.auto_mode = 1
                        
                        # Save operation data
                        data_processor.save_operation_data(self.record_file, operation_dict)
                        
                        # Send movement command if changed
                        new_instruction = operation_dict['移动操作']
                        if new_instruction != self.old_instruction and new_instruction != '无移动':
                            self.old_instruction = new_instruction
                            logger.debug(f"Manual mode, sending movement: {self.old_instruction}")
                            try:
                                game_controller.send_movement(self.old_instruction)
                            except Exception as e:
                                logger.error(f"Failed to send movement: {e}")
                                self.running = False
                                break
                            
                            time.sleep(0.01)
                        
                        # Send action command
                        if operation_dict['动作操作'] != '无动作' and operation_dict['动作操作'] not in ['发起集合', '发起进攻', '发起撤退']:
                            logger.debug(f"Manual mode, sending action: {operation_dict['动作操作']}")
                            try:
                                game_controller.send_command(operation_dict['动作操作'])
                            except Exception as e:
                                logger.error(f"Failed to send action: {e}")
                                self.running = False
                                break
                    else:
                        # AI mode
                        game_controller.operation_queue = []
                        operation_dict['移动操作'] = instruction_parts[0]
                        operation_dict['动作操作'] = instruction_parts[1]
                        
                        # Send movement command if changed
                        new_instruction = instruction_parts[0]
                        if new_instruction != self.old_instruction and new_instruction != '无移动':
                            self.old_instruction = new_instruction
                            logger.debug(f"AI mode, sending movement: {self.old_instruction}")
                            try:
                                game_controller.send_movement(self.old_instruction)
                            except Exception as e:
                                logger.error(f"Failed to send movement: {e}")
                                self.running = False
                                break
                            
                            time.sleep(0.01)
                        
                        # Save image
                        img_path = data_processor.save_image(img, self.session_dir, i)
                        
                        # Set end flag
                        self.auto_mode = 0
                        operation_dict['结束'] = 0
                        
                        # Save operation data
                        data_processor.save_operation_data(self.record_file, operation_dict)
                        
                        # Send action command
                        if instruction_parts[1] != '无动作' and instruction_parts[1] not in ['发起集合', '发起进攻', '发起撤退']:
                            logger.debug(f"AI mode, sending action: {instruction_parts[1]}")
                            try:
                                game_controller.send_command(instruction_parts[1])
                            except Exception as e:
                                logger.error(f"Failed to send action: {e}")
                                self.running = False
                                break
                
                # Sleep to maintain frame rate
                iter_time = time.time() - iter_start
                sleep_time = self.screenshot_interval - iter_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Increment counter
                self.count += 1
                
                # Play notification sound at intervals
                if i % 3000 == 0 and i > 0:
                    logger.info(f"Reached {i} iterations")
        
        except Exception as e:
            logger.exception(f"Error in game session: {e}")
        finally:
            # Ensure record file is closed
            if self.record_file:
                self.record_file.close()
                self.record_file = None
            
            logger.info("Game session ended")
            self.running = False

def start_scrcpy():
    """Start the scrcpy process."""
    try:
        # Kill existing processes
        os.system('taskkill /IM scrcpy.exe /F')
        os.system('taskkill /IM adb.exe /F')
        
        # Start scrcpy
        os.system("scrcpy --max-size 960")
        logger.info("Started scrcpy")
        return True
    except Exception as e:
        logger.error(f"Failed to start scrcpy: {e}")
        return False

def process_training_data():
    """Process all unprocessed training data."""
    try:
        count = data_processor.process_all_unprocessed_sessions()
        logger.info(f"Processed {count} training data sessions")
        return count
    except Exception as e:
        logger.error(f"Failed to process training data: {e}")
        return 0

def train_model(epochs=1):
    """Train the model on processed data."""
    try:
        from 训练X import train_model as train_model_impl
        train_model_impl(epochs)
        logger.info(f"Trained model for {epochs} epochs")
        return True
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WZCQ AI Game Player")
    parser.add_argument("--mode", choices=["play", "train", "process", "schedule"], default="play",
                        help="Mode to run in: play, train, process, or schedule")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration in seconds to run the session")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to train for")
    parser.add_argument("--schedule", type=str, default=None,
                        help="Schedule time in HH:MM format")
    parser.add_argument("--script", type=str, default="main.py",
                        help="Script to schedule")
    
    args = parser.parse_args()
    
    if args.mode == "play":
        # Start scrcpy
        start_scrcpy()
        
        # Start game session
        session = GameSession()
        if args.duration:
            config.set("automation", "session_duration", args.duration)
        
        try:
            session.start()
            
            # Wait for session to end
            while session.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            session.stop()
    
    elif args.mode == "train":
        # Train model
        train_model(args.epochs)
    
    elif args.mode == "process":
        # Process training data
        process_training_data()
    
    elif args.mode == "schedule":
        if args.schedule:
            # Add scheduled run
            scheduler.add_scheduled_run(
                args.schedule, args.script, args.duration,
                {"mode": "play", "duration": args.duration}
            )
            logger.info(f"Added scheduled run for {args.script} at {args.schedule}")
        
        # Start scheduler
        scheduler.start()
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            scheduler.stop()

if __name__ == "__main__":
    main()