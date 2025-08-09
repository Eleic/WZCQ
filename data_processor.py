"""
Data processor module for the WZCQ project.
This module handles data collection, preprocessing, and storage.
"""

import os
import time
import json
import numpy as np
import torch
import torchvision
from PIL import Image
from pathlib import Path

from config_manager import config
from logger import logger
from resnet_utils import myResnet
from 杂项 import 读出引索

class DataProcessor:
    """Handles data processing for the WZCQ project."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.training_data_dir = config.get("paths", "training_data_dir")
        Path(self.training_data_dir).mkdir(parents=True, exist_ok=True)
        
        # Load dictionaries
        self.词数词典路径 = os.path.join(config.get("paths", "json_dir"), "词_数表.json")
        self.数_词表路径 = os.path.join(config.get("paths", "json_dir"), "数_词表.json")
        
        self.词_数表, self.数_词表 = 读出引索(self.词数词典路径, self.数_词表路径)
        
        with open(self.词数词典路径, encoding='utf8') as f:
            self.词数词典 = json.load(f)
        
        # Initialize device and model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.resnet101 = self._initialize_resnet()
    
    def _initialize_resnet(self):
        """Initialize the ResNet model."""
        try:
            model = torchvision.models.resnet101(pretrained=True).eval()
            model = myResnet(model).to(self.device).requires_grad_(False)
            logger.info("ResNet model initialized successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize ResNet model: {e}")
            raise
    
    def create_session_directory(self):
        """Create a new directory for the current session."""
        session_dir = os.path.join(self.training_data_dir, str(int(time.time())))
        os.makedirs(session_dir, exist_ok=True)
        logger.info(f"Created session directory: {session_dir}")
        return session_dir
    
    def preprocess_image(self, image):
        """Preprocess an image using ResNet."""
        try:
            img_array = np.array(image)
            img_tensor = torch.from_numpy(img_array).to(self.device).unsqueeze(0).permute(0, 3, 2, 1) / 255
            _, features = self.resnet101(img_tensor)
            return features.reshape(1, 6*6*2048)
        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            raise
    
    def save_image(self, image, path, index):
        """Save an image to disk."""
        try:
            image_path = os.path.join(path, f"{index}.jpg")
            image.save(image_path)
            return image_path
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise
    
    def save_operation_data(self, file_handle, data):
        """Save operation data to a JSON file."""
        try:
            json.dump(data, file_handle, ensure_ascii=False)
            file_handle.write('\n')
        except Exception as e:
            logger.error(f"Failed to save operation data: {e}")
            raise
    
    def process_training_data(self, session_dir):
        """Process training data for a session."""
        try:
            json_path = os.path.join(session_dir, "_操作数据.json")
            npz_path = os.path.join(session_dir, "图片_操作预处理数据2.npz")
            
            # Skip if already processed
            if os.path.exists(npz_path):
                logger.info(f"Session {session_dir} already processed, skipping")
                return
            
            logger.info(f"Processing training data for session {session_dir}")
            
            # Initialize tensors
            图片张量 = torch.Tensor(0)
            操作序列 = np.ones((1, 1))
            结束序列 = np.ones((1, 1))
            
            # Load operation data
            data_list = []
            with open(json_path, encoding='ansi') as f:
                for line in f:
                    if not line.strip():
                        continue
                    line = line.replace("'", '"')
                    data = json.loads(line)
                    data_list.append(data)
            
            # Process each data entry
            移动操作 = '无移动'
            for i, data in enumerate(data_list):
                img_path = os.path.join(session_dir, f"{data['图片号']}.jpg")
                img = Image.open(img_path)
                
                # Process first image
                if 图片张量.shape[0] == 0:
                    img_tensor = np.array(img)
                    img_tensor = torch.from_numpy(img_tensor).to(self.device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                    _, out = self.resnet101(img_tensor)
                    图片张量 = out.reshape(1, 6*6*2048)
                    
                    移动操作a = data["移动操作"]
                    if 移动操作a != '无移动':
                        移动操作 = 移动操作a
                    
                    操作序列[0, 0] = self.词数词典[移动操作 + "_" + data["动作操作"]]
                    结束序列[0, 0] = data["结束"]
                else:
                    img_tensor = np.array(img)
                    img_tensor = torch.from_numpy(img_tensor).to(self.device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                    _, out = self.resnet101(img_tensor)
                    
                    图片张量 = torch.cat((图片张量, out.reshape(1, 6*6*2048)), 0)
                    移动操作a = data["移动操作"]
                    if 移动操作a != '无移动':
                        移动操作 = 移动操作a
                    
                    操作序列 = np.append(操作序列, self.词数词典[移动操作 + "_" + data["动作操作"]])
                    结束序列 = np.append(结束序列, data["结束"])
            
            # Convert to numpy and save
            图片张量np = 图片张量.cpu().numpy()
            操作序列 = 操作序列.astype(np.int64)
            np.savez(npz_path, 图片张量np=图片张量np, 操作序列=操作序列, 结束序列=结束序列)
            
            logger.info(f"Successfully processed training data for session {session_dir}")
            return npz_path
        except Exception as e:
            logger.error(f"Failed to process training data for session {session_dir}: {e}")
            raise
    
    def process_all_unprocessed_sessions(self):
        """Process all unprocessed training data sessions."""
        try:
            processed_count = 0
            for session_dir in os.listdir(self.training_data_dir):
                session_path = os.path.join(self.training_data_dir, session_dir)
                if not os.path.isdir(session_path):
                    continue
                
                json_path = os.path.join(session_path, "_操作数据.json")
                npz_path = os.path.join(session_path, "图片_操作预处理数据2.npz")
                
                if os.path.exists(json_path) and not os.path.exists(npz_path):
                    self.process_training_data(session_path)
                    processed_count += 1
            
            logger.info(f"Processed {processed_count} unprocessed sessions")
            return processed_count
        except Exception as e:
            logger.error(f"Failed to process unprocessed sessions: {e}")
            raise

# Create a singleton instance
data_processor = DataProcessor()