"""
Training module for the WZCQ project.
This module handles model training.
"""

import os
import time
import torch
import numpy as np
import random
from Batch import create_masks

from config_manager import config
from logger import logger
from ai_agent import ai_agent
from 模型_策略梯度 import Transformer

def train_model(epochs=1):
    """Train the model on processed data."""
    try:
        # Get configuration
        training_data_dir = config.get("paths", "training_data_dir")
        chunk_size = config.get("training", "chunk_size")
        cursor_size = config.get("training", "cursor_size")
        branches = config.get("training", "branches")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Get all session directories
        session_dirs = []
        for root, dirs, files in os.walk(training_data_dir):
            if len(dirs) > 0:
                session_dirs = dirs
                break
        
        # Shuffle directories for better training
        random.shuffle(session_dirs)
        
        # Initialize state model
        state_model = ai_agent.state_model
        state_list = ai_agent.state_list
        state_dict = ai_agent.state_dict
        
        # Training loop
        count = 0
        time_start = time.time()
        
        for epoch in range(epochs):
            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            
            for session_dir in session_dirs:
                # Load preprocessed data
                npz_path = os.path.join(training_data_dir, session_dir, "图片_操作预处理数据2.npz")
                if not os.path.isfile(npz_path):
                    logger.warning(f"Preprocessed data not found for session {session_dir}")
                    continue
                
                try:
                    npz_file = np.load(npz_path, allow_pickle=True)
                    image_tensor_np, operation_sequence = npz_file["图片张量np"], npz_file["操作序列"]
                    
                    # Skip if too small
                    if image_tensor_np.shape[0] < chunk_size:
                        logger.warning(f"Session {session_dir} has too few samples ({image_tensor_np.shape[0]}), skipping")
                        continue
                    
                    # Process data in chunks
                    cursor = 0
                    operation_sequence = np.insert(operation_sequence, 0, 128)
                    
                    operation_chunks = []
                    target_chunks = []
                    image_chunks = []
                    
                    # Split data into chunks
                    while cursor + chunk_size < operation_sequence.shape[0]:
                        operation_chunk = operation_sequence[cursor:cursor + chunk_size]
                        target_chunk = operation_sequence[cursor + 1:cursor + 1 + chunk_size]
                        image_chunk = image_tensor_np[cursor:cursor + chunk_size, :]
                        
                        operation_chunks.append(operation_chunk)
                        target_chunks.append(target_chunk)
                        image_chunks.append(image_chunk)
                        
                        cursor += cursor_size
                    
                    # Add final chunk
                    operation_chunk = operation_sequence[-chunk_size - 1:-1]
                    target_chunk = operation_sequence[-chunk_size:]
                    image_chunk = image_tensor_np[-chunk_size:, :]
                    
                    operation_chunks.append(operation_chunk)
                    target_chunks.append(target_chunk)
                    image_chunks.append(image_chunk)
                    
                    # Process chunks in branches
                    branch_idx = 0
                    while branch_idx * branches < len(operation_chunks):
                        end_idx = min((branch_idx + 1) * branches, len(operation_chunks))
                        
                        # Get branch data
                        operation_branch = np.array(operation_chunks[branch_idx * branches:end_idx])
                        image_branch = np.array(image_chunks[branch_idx * branches:end_idx], dtype=np.float32)
                        target_branch = np.array(target_chunks[branch_idx * branches:end_idx])
                        
                        # Convert to torch tensors
                        operation_branch_torch = torch.from_numpy(operation_branch).to(device)
                        operation_dummy = np.ones_like(operation_branch)
                        operation_dummy_torch = torch.from_numpy(operation_dummy).to(device)
                        image_branch_torch = torch.from_numpy(image_branch).to(device)
                        target_branch_torch = torch.from_numpy(target_branch).to(device)
                        
                        # Create masks
                        src_mask, trg_mask = create_masks(operation_branch_torch, operation_branch_torch, device)
                        
                        # Skip if shapes don't match
                        if image_branch_torch.shape[0] != operation_branch_torch.shape[0]:
                            logger.warning(f"Shape mismatch in session {session_dir}, branch {branch_idx}")
                            branch_idx += 1
                            continue
                        
                        # Prepare state
                        state = {
                            'operation_sequence': operation_branch,
                            'image_tensor': image_branch,
                            'trg_mask': trg_mask
                        }
                        
                        # Select actions
                        actions, action_probs, evaluation = ai_agent.agent.选择动作批量(
                            state, device, target_branch_torch, True
                        )
                        
                        # Evaluate state
                        output, _ = state_model(image_branch_torch, operation_dummy_torch, trg_mask)
                        _, samples = torch.topk(output, k=1, dim=-1)
                        samples_np = samples.cpu().numpy()
                        
                        # Calculate rewards
                        rewards = np.ones_like(samples_np[0, :, 0], dtype=np.float32)
                        for i in range(samples_np.shape[1]):
                            state_name = state_list[samples_np[0, i, 0]]
                            reward = state_dict[state_name]
                            rewards[i] = reward
                        
                        # Train agent
                        ai_agent.train_batch(state, rewards, actions, action_probs, evaluation)
                        
                        # Log progress
                        if count % 10 == 0:
                            time_end = time.time()
                            elapsed = time_end - time_start
                            logger.info(f"Epoch {epoch+1}/{epochs}, Session {session_dir}, Branch {branch_idx}, Count {count}, Time {elapsed:.2f}s")
                        
                        count += 1
                        branch_idx += 1
                
                except Exception as e:
                    logger.error(f"Error processing session {session_dir}: {e}")
                    continue
            
            # Save model after each epoch
            ai_agent.save_model(epoch)
            logger.info(f"Completed epoch {epoch+1}/{epochs}")
        
        logger.info(f"Training completed after {epochs} epochs")
        return True
    
    except Exception as e:
        logger.exception(f"Error in train_model: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the WZCQ model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    
    args = parser.parse_args()
    
    train_model(args.epochs)