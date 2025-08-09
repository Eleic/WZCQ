"""
Game controller module for the WZCQ project.
This module provides a unified interface for controlling the game.
"""

import os
import time
import threading
from pynput.keyboard import Controller, Key, Listener
from pynput import keyboard
from PIL import Image
import numpy as np
import torch

from config_manager import config
from logger import logger
from 取训练数据 import MyMNTDevice
from 杂项 import 读出引索

class GameController:
    """Controls the game through keyboard and touch inputs."""
    
    def __init__(self):
        """Initialize the game controller."""
        self.device_id = config.get("device", "device_id")
        self.window_name = config.get("device", "window_name")
        self.device = MyMNTDevice(self.device_id)
        
        # Load operation dictionaries
        self.词数词典路径 = os.path.join(config.get("paths", "json_dir"), "词_数表.json")
        self.数_词表路径 = os.path.join(config.get("paths", "json_dir"), "数_词表.json")
        self.操作查询路径 = os.path.join(config.get("paths", "json_dir"), "名称_操作.json")
        
        self.词_数表, self.数_词表 = 读出引索(self.词数词典路径, self.数_词表路径)
        
        with open(self.操作查询路径, encoding='utf8') as f:
            import json
            self.操作查询词典 = json.load(f)
        
        # Key state tracking
        self.key_states = {
            'w': False, 'a': False, 's': False, 'd': False, 'q': False,
            'attack': False, 'manual_mode': False, 'ai_enabled': True
        }
        
        # Operation queue
        self.operation_queue = []
        self.lock = threading.Lock()
        
        # Predefined touch commands
        self.commands = {
            'add_skill_3': 'd 0 552 1878 100\nc\nu 0\nc\n',
            'add_skill_2': 'd 0 446 1687 100\nc\nu 0\nc\n',
            'add_skill_1': 'd 0 241 1559 100\nc\nu 0\nc\n',
            'purchase': 'd 0 651 207 100\nc\nu 0\nc\n'
        }
        
        # Start keyboard listener
        self.listener = None
    
    def start_keyboard_listener(self):
        """Start the keyboard listener thread."""
        if self.listener is None or not self.listener.is_alive():
            self.listener = Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self.listener.daemon = True
            self.listener.start()
            logger.info("Keyboard listener started")
    
    def stop_keyboard_listener(self):
        """Stop the keyboard listener thread."""
        if self.listener and self.listener.is_alive():
            self.listener.stop()
            logger.info("Keyboard listener stopped")
    
    def _get_key_name(self, key):
        """Get the name of a key."""
        if isinstance(key, keyboard.KeyCode):
            return key.char
        else:
            return str(key)
    
    def _on_press(self, key):
        """Handle key press events."""
        key_name = self._get_key_name(key)
        operation = ''
        
        # Movement keys
        if key_name == 'w':
            self.key_states['w'] = True
        elif key_name == 'a':
            self.key_states['a'] = True
        elif key_name == 's':
            self.key_states['s'] = True
        elif key_name == 'd':
            self.key_states['d'] = True
        elif key_name == 'q':
            self.key_states['q'] = True
        elif key_name == 'i':
            self.key_states['ai_enabled'] = not self.key_states['ai_enabled']
            logger.info(f"AI {'enabled' if self.key_states['ai_enabled'] else 'disabled'}")
        
        # Action keys
        elif key_name == 'Key.space':
            operation = '召唤师技能'
        elif key_name == 'Key.end':
            operation = '补刀'
        elif key_name == 'Key.page_down':
            operation = '推塔'
        elif key_name == 'j' or key_name == 'Key.left':
            operation = '一技能'
        elif key_name == 'k' or key_name == 'Key.down':
            operation = '二技能'
        elif key_name == 'l' or key_name == 'Key.right':
            operation = '三技能'
        elif key_name == 'f':
            operation = '回城'
        elif key_name == 'g':
            operation = '恢复'
        elif key_name == 'h':
            operation = '召唤师技能'
        elif key_name == 'Key.up':
            self.key_states['attack'] = True
        
        # Add operation to queue if not empty
        with self.lock:
            if operation:
                self.operation_queue.append(operation)
    
    def _on_release(self, key):
        """Handle key release events."""
        key_name = self._get_key_name(key)
        
        # Movement keys
        if key_name == 'w':
            self.key_states['w'] = False
        elif key_name == 'a':
            self.key_states['a'] = False
        elif key_name == 's':
            self.key_states['s'] = False
        elif key_name == 'd':
            self.key_states['d'] = False
        elif key_name == 'q':
            self.key_states['q'] = False
        elif key_name == 'Key.up':
            self.key_states['attack'] = False
        
        # Exit on escape key
        if key == Key.esc:
            return False
    
    def get_movement_direction(self):
        """Get the current movement direction based on key states."""
        if self.key_states['q']:
            return '移动停'
        elif self.key_states['w'] and not self.key_states['s'] and not self.key_states['a'] and not self.key_states['d']:
            return '上移'
        elif not self.key_states['w'] and self.key_states['s'] and not self.key_states['a'] and not self.key_states['d']:
            return '下移'
        elif not self.key_states['w'] and not self.key_states['s'] and self.key_states['a'] and not self.key_states['d']:
            return '左移'
        elif not self.key_states['w'] and not self.key_states['s'] and not self.key_states['a'] and self.key_states['d']:
            return '右移'
        elif self.key_states['w'] and not self.key_states['s'] and self.key_states['a'] and not self.key_states['d']:
            return '左上移'
        elif self.key_states['w'] and not self.key_states['s'] and not self.key_states['a'] and self.key_states['d']:
            return '右上移'
        elif not self.key_states['w'] and self.key_states['s'] and self.key_states['a'] and not self.key_states['d']:
            return '左下移'
        elif not self.key_states['w'] and self.key_states['s'] and not self.key_states['a'] and self.key_states['d']:
            return '右下移'
        else:
            return ''
    
    def get_next_operation(self):
        """Get the next operation from the queue."""
        with self.lock:
            if self.operation_queue:
                return self.operation_queue.pop(0)
            return None
    
    def send_command(self, command_name):
        """Send a predefined command to the device."""
        try:
            if command_name in self.commands:
                self.device.发送(self.commands[command_name])
                logger.debug(f"Sent command: {command_name}")
                return True
            elif command_name in self.操作查询词典:
                self.device.发送(self.操作查询词典[command_name])
                logger.debug(f"Sent operation: {command_name}")
                return True
            else:
                logger.warning(f"Unknown command: {command_name}")
                return False
        except Exception as e:
            logger.error(f"Failed to send command {command_name}: {e}")
            return False
    
    def send_movement(self, direction):
        """Send a movement command to the device."""
        try:
            if direction in self.操作查询词典:
                self.device.发送(self.操作查询词典[direction])
                logger.debug(f"Sent movement: {direction}")
                return True
            else:
                logger.warning(f"Unknown direction: {direction}")
                return False
        except Exception as e:
            logger.error(f"Failed to send movement {direction}: {e}")
            return False
    
    def perform_periodic_actions(self):
        """Perform periodic actions like buying items and upgrading skills."""
        try:
            self.send_command('purchase')
            time.sleep(0.02)
            self.send_command('add_skill_3')
            time.sleep(0.02)
            self.send_command('add_skill_2')
            time.sleep(0.02)
            self.send_command('add_skill_1')
            time.sleep(0.02)
            logger.info("Performed periodic actions")
            return True
        except Exception as e:
            logger.error(f"Failed to perform periodic actions: {e}")
            return False
    
    def is_ai_enabled(self):
        """Check if AI is enabled."""
        return self.key_states['ai_enabled']
    
    def is_attack_active(self):
        """Check if attack is active."""
        return self.key_states['attack']

# Create a singleton instance
game_controller = GameController()