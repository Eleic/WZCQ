"""
AI agent module for the WZCQ project.
This module handles the AI logic for playing the game.
"""

import os
import torch
import numpy as np
from Batch import create_masks
from 辅助功能 import 状态信息综合

from config_manager import config
from logger import logger
from 模型_策略梯度 import 智能体, Transformer

class AIAgent:
    """AI agent for playing the game."""
    
    def __init__(self):
        """Initialize the AI agent."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load model configurations
        self.iterations = config.get("training", "iterations")
        self.batch_size = config.get("training", "batch_size")
        self.epochs = config.get("training", "epochs")
        self.learning_rate = config.get("training", "learning_rate")
        self.input_dim = config.get("training", "input_dim")
        
        # Initialize models
        self.agent = self._initialize_agent()
        self.state_model = self._initialize_state_model()
        
        # State dictionary for reward mapping
        self.state_dict = {
            '击杀小兵或野怪或推掉塔': 2,
            '击杀敌方英雄': 5,
            '被击塔攻击': -0.5,
            '被击杀': -2,
            '无状况': 0.01,
            '死亡': 0.01,
            '其它': -0.003,
            '普通': 0.01
        }
        
        self.state_dict_b = {
            '击杀小兵或野怪或推掉塔': 0,
            '击杀敌方英雄': 1,
            '被击塔攻击': 2,
            '被击杀': 3,
            '死亡': 4,
            '普通': 5
        }
        
        self.state_list = list(self.state_dict_b.keys())
    
    def _initialize_agent(self):
        """Initialize the agent model."""
        try:
            agent = 智能体(
                动作数=7,
                并行条目数=self.batch_size,
                学习率=self.learning_rate,
                轮数=self.epochs,
                输入维度=self.input_dim
            )
            
            # Load weights if available
            weights_path = os.path.join(config.get("paths", "weights_dir"), "model_weights_2021-05-7D")
            if os.path.exists(weights_path):
                agent.加载模型(weights_path)
                logger.info(f"Loaded agent model from {weights_path}")
            else:
                logger.warning(f"Agent model weights not found at {weights_path}")
            
            return agent
        except Exception as e:
            logger.error(f"Failed to initialize agent model: {e}")
            raise
    
    def _initialize_state_model(self):
        """Initialize the state judgment model."""
        try:
            model = Transformer(6, 768, 2, 12, 0.0, 6*6*2048)
            
            # Load weights if available
            weights_path = os.path.join(config.get("paths", "weights_dir"), "model_weights_判断状态L")
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path))
                model = model.to(self.device).requires_grad_(False)
                logger.info(f"Loaded state model from {weights_path}")
            else:
                logger.warning(f"State model weights not found at {weights_path}")
            
            return model
        except Exception as e:
            logger.error(f"Failed to initialize state model: {e}")
            raise
    
    def select_action(self, image_tensor, operation_sequence, trg_mask):
        """Select an action based on the current state."""
        try:
            # Prepare state
            state = 状态信息综合(image_tensor.cpu().numpy(), operation_sequence, trg_mask)
            
            # Select action
            action, action_probs, evaluation = self.agent.选择动作(state, self.device, 1, False)
            
            return action, action_probs, evaluation
        except Exception as e:
            logger.error(f"Failed to select action: {e}")
            raise
    
    def evaluate_state(self, image_tensor, operation_sequence, trg_mask):
        """Evaluate the current game state."""
        try:
            # Create dummy operation sequence for state model
            dummy_ops = np.ones_like(operation_sequence)
            dummy_ops_tensor = torch.from_numpy(dummy_ops).to(self.device)
            
            # Get state evaluation
            output, _ = self.state_model(image_tensor, dummy_ops_tensor, trg_mask)
            _, samples = torch.topk(output, k=1, dim=-1)
            samples_np = samples.cpu().numpy()
            
            # Convert to rewards
            rewards = np.ones_like(samples_np[0, :, 0], dtype=np.float32)
            
            for i in range(samples_np.shape[1]):
                state_name = self.state_list[samples_np[0, i, 0]]
                reward = self.state_dict[state_name]
                rewards[i] = reward
            
            return rewards, samples_np
        except Exception as e:
            logger.error(f"Failed to evaluate state: {e}")
            raise
    
    def train_batch(self, state, rewards, actions, action_probs, evaluation):
        """Train the agent on a batch of data."""
        try:
            self.agent.监督强化学习(self.device, state, rewards, actions, action_probs, evaluation)
            logger.info("Trained agent on batch")
            return True
        except Exception as e:
            logger.error(f"Failed to train agent: {e}")
            return False
    
    def save_model(self, epoch):
        """Save the agent model."""
        try:
            self.agent.保存模型(epoch)
            logger.info(f"Saved agent model for epoch {epoch}")
            return True
        except Exception as e:
            logger.error(f"Failed to save agent model: {e}")
            return False

# Create a singleton instance
ai_agent = AIAgent()