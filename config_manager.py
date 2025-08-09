"""
Centralized configuration manager for the WZCQ project.
This module provides a unified way to manage all configuration parameters.
"""

import os
import json
import yaml
import logging
from pathlib import Path

class ConfigManager:
    """Manages all configuration settings for the WZCQ project."""
    
    DEFAULT_CONFIG = {
        # Device settings
        "device": {
            "device_id": "68UDU17B14011947",
            "window_name": "RNE-AL00",
            "screen_resolution": [1080, 2160]
        },
        
        # Path settings
        "paths": {
            "training_data_dir": "../训练数据样本/未用",
            "weights_dir": "./weights",
            "json_dir": "./json",
            "logs_dir": "./logs"
        },
        
        # Model settings
        "model": {
            "main_model_path": "weights/model_weights_2021-05-7D",
            "state_model_path": "weights/model_weights_判断状态L",
            "save_interval": 1,
            "batch_size": 100,
            "learning_rate": 0.0003,
            "epochs": 3
        },
        
        # Training settings
        "training": {
            "iterations": 15000,
            "batch_size": 100,
            "epochs": 3,
            "learning_rate": 0.0003,
            "input_dim": 6,
            "chunk_size": 600,
            "cursor_size": 600,
            "branches": 1
        },
        
        # Game control settings
        "controls": {
            "move_keys": ["w", "a", "s", "d"],
            "skill_keys": ["Key.left", "Key.down", "Key.right"],
            "attack_key": "Key.up",
            "toggle_ai_key": "i",
            "exit_key": "Key.esc"
        },
        
        # Automation settings
        "automation": {
            "auto_restart": True,
            "auto_reconnect": True,
            "session_duration": 3000,
            "screenshot_interval": 0.22,
            "max_sessions": 10,
            "scheduled_runs": []
        }
    }
    
    def __init__(self, config_path="./config.yaml"):
        """Initialize the configuration manager."""
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_directories()
        
    def _load_config(self):
        """Load configuration from file or create default if not exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith('.json'):
                        return json.load(f)
                    else:
                        return yaml.safe_load(f)
            except Exception as e:
                logging.error(f"Error loading config: {e}")
                return self.DEFAULT_CONFIG
        else:
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG
    
    def _save_config(self, config):
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.endswith('.json'):
                    json.dump(config, f, ensure_ascii=False, indent=4)
                else:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logging.error(f"Error saving config: {e}")
    
    def _setup_directories(self):
        """Ensure all required directories exist."""
        for path_key, path_value in self.config["paths"].items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def get(self, section, key=None):
        """Get a configuration value."""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def set(self, section, key, value):
        """Set a configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self._save_config(self.config)
    
    def update(self, new_config):
        """Update multiple configuration values at once."""
        self.config.update(new_config)
        self._save_config(self.config)
    
    def add_scheduled_run(self, time, duration, params=None):
        """Add a scheduled run to the automation settings."""
        if "scheduled_runs" not in self.config["automation"]:
            self.config["automation"]["scheduled_runs"] = []
        
        self.config["automation"]["scheduled_runs"].append({
            "time": time,
            "duration": duration,
            "params": params or {}
        })
        self._save_config(self.config)

# Create a singleton instance
config = ConfigManager()