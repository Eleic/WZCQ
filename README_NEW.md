# WZCQ - AI for Honor of Kings (王者荣耀)

This project uses reinforcement learning to train an AI to play the mobile game "Honor of Kings" (王者荣耀). The AI uses computer vision to understand the game state and makes decisions based on a trained model.

## Optimized Architecture

The codebase has been optimized with the following improvements:

1. **Centralized Configuration**: All settings are managed through a single configuration system.
2. **Modular Architecture**: Code is organized into logical modules with clear responsibilities.
3. **Improved Error Handling**: Robust error handling and recovery mechanisms.
4. **Logging System**: Comprehensive logging for better debugging and monitoring.
5. **Automated Execution**: Scheduler for running tasks automatically.
6. **Optimized Data Processing**: More efficient data collection and preprocessing.

## Requirements

- Python 3.8+
- CUDA-compatible GPU (GTX 1060 or better recommended)
- Android device with USB debugging enabled
- scrcpy for screen mirroring
- minitouch for touch input

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/WZCQ.git
   cd WZCQ
   ```

2. Install dependencies:
   ```
   pip install -r requirements_new.txt
   ```

3. Download scrcpy and extract it to the project root directory.

4. Download the pre-trained models:
   - Main model: [Google Drive](https://drive.google.com/file/d/10NXGuEUYuRJyQvPN1kXxkBekoar3gwME/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1ZMCierCZkskEsgjj_wwwyw) (Code: oiar)
   - State model: [Google Drive](https://drive.google.com/file/d/1eqy-xX29sjEguuQI_1m8qaLEX3g4KAQ7/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1-UCuPutZQck3Iawot9bGrw) (Code: 545t)

5. Place the downloaded models in the `weights` directory.

## Usage

### Running the AI

1. Start the game session:
   ```
   python main.py --mode play
   ```

2. The AI will start playing the game. You can take control at any time by pressing the movement keys (W, A, S, D) or action keys.

3. Press 'i' to toggle AI control on/off.

4. Press 'Esc' to exit.

### Training the Model

1. Collect training data by playing the game:
   ```
   python main.py --mode play
   ```

2. Process the collected data:
   ```
   python main.py --mode process
   ```

3. Train the model:
   ```
   python main.py --mode train --epochs 3
   ```

### Scheduling Automated Sessions

1. Schedule a session to run at a specific time:
   ```
   python main.py --mode schedule --schedule "22:00" --duration 3600
   ```

2. This will run the AI for 1 hour starting at 10:00 PM.

## Configuration

All settings can be configured in the `config.yaml` file. The main sections are:

- `device`: Settings for the Android device
- `paths`: File paths for data, models, etc.
- `model`: Model configuration
- `training`: Training parameters
- `controls`: Keyboard controls
- `automation`: Settings for automated execution

## Key Controls

- **W, A, S, D**: Movement
- **J, K, L** or **Left, Down, Right arrows**: Skills 1, 2, 3
- **Up arrow**: Attack
- **Space**: Summoner spell
- **F**: Recall
- **G**: Heal
- **H**: Summoner spell
- **I**: Toggle AI on/off
- **Esc**: Exit

## Automated Execution

The scheduler allows you to run the AI automatically at specified times. This is useful for:

1. **Collecting Training Data**: Schedule sessions to collect data while you're away.
2. **Continuous Training**: Automatically train the model with new data.
3. **Regular Play**: Have the AI play at specific times.

To set up a scheduled task:

```python
from scheduler import scheduler

# Schedule a 1-hour play session at 10:00 PM
scheduler.add_scheduled_run("22:00", "main.py", 3600, {"mode": "play"})

# Start the scheduler
scheduler.start()
```

## Troubleshooting

- **Screen Capture Issues**: Make sure scrcpy is properly installed and the Android device is connected with USB debugging enabled.
- **Touch Input Issues**: Check that minitouch is working correctly. For Android 10+, you may need alternative input methods.
- **Model Loading Errors**: Verify that the model files are in the correct location and have the right format.
- **Performance Issues**: Adjust the screenshot interval in the configuration to match your hardware capabilities.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is based on the original [AI for Honor of Kings](https://github.com/FengQuanLi/ResnetGPT) project.
- Thanks to the developers of scrcpy and minitouch for their excellent tools.