"""
Scheduler module for the WZCQ project.
This module handles scheduling of automated tasks.
"""

import os
import time
import datetime
import threading
import schedule
import signal
import subprocess
from pathlib import Path

from config_manager import config
from logger import logger

class Scheduler:
    """Handles scheduling of automated tasks."""
    
    def __init__(self):
        """Initialize the scheduler."""
        self.running_processes = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.scheduler_thread = None
    
    def _run_process(self, script_name, duration=None, params=None):
        """Run a process with optional duration and parameters."""
        try:
            script_path = os.path.join(os.getcwd(), script_name)
            
            # Prepare command
            cmd = ["python", script_path]
            if params:
                for key, value in params.items():
                    cmd.extend([f"--{key}", str(value)])
            
            # Start process
            process = subprocess.Popen(cmd)
            process_id = process.pid
            
            with self.lock:
                self.running_processes[process_id] = {
                    "process": process,
                    "script": script_name,
                    "start_time": time.time(),
                    "duration": duration
                }
            
            logger.info(f"Started process {process_id} for script {script_name}")
            
            # If duration is specified, schedule termination
            if duration:
                def terminate_process():
                    with self.lock:
                        if process_id in self.running_processes:
                            process = self.running_processes[process_id]["process"]
                            if process.poll() is None:  # Process is still running
                                process.terminate()
                                logger.info(f"Terminated process {process_id} after {duration} seconds")
                            del self.running_processes[process_id]
                
                threading.Timer(duration, terminate_process).start()
            
            return process_id
        except Exception as e:
            logger.error(f"Failed to run process for script {script_name}: {e}")
            return None
    
    def _check_processes(self):
        """Check and clean up finished processes."""
        with self.lock:
            to_remove = []
            current_time = time.time()
            
            for pid, info in self.running_processes.items():
                process = info["process"]
                
                # Check if process has finished
                if process.poll() is not None:
                    to_remove.append(pid)
                    logger.info(f"Process {pid} for script {info['script']} has finished with return code {process.returncode}")
                
                # Check if process has exceeded its duration
                elif info["duration"] and (current_time - info["start_time"]) > info["duration"]:
                    process.terminate()
                    to_remove.append(pid)
                    logger.info(f"Terminated process {pid} for script {info['script']} after {info['duration']} seconds")
            
            # Remove finished processes
            for pid in to_remove:
                del self.running_processes[pid]
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self.stop_event.is_set():
            schedule.run_pending()
            self._check_processes()
            time.sleep(1)
    
    def start(self):
        """Start the scheduler."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Scheduler is already running")
            return False
        
        # Clear existing jobs
        schedule.clear()
        
        # Load scheduled runs from config
        scheduled_runs = config.get("automation", "scheduled_runs")
        if scheduled_runs:
            for run in scheduled_runs:
                time_str = run["time"]
                script = run.get("script", "main.py")
                duration = run.get("duration")
                params = run.get("params")
                
                # Schedule the job
                schedule.every().day.at(time_str).do(
                    self._run_process, script, duration, params
                )
                logger.info(f"Scheduled {script} to run daily at {time_str}")
        
        # Start scheduler thread
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Scheduler started")
        return True
    
    def stop(self):
        """Stop the scheduler."""
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            logger.warning("Scheduler is not running")
            return False
        
        # Stop scheduler thread
        self.stop_event.set()
        self.scheduler_thread.join(timeout=5)
        
        # Terminate all running processes
        with self.lock:
            for pid, info in self.running_processes.items():
                process = info["process"]
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    logger.info(f"Terminated process {pid} for script {info['script']}")
            
            self.running_processes.clear()
        
        logger.info("Scheduler stopped")
        return True
    
    def add_scheduled_run(self, time_str, script, duration=None, params=None):
        """Add a scheduled run."""
        try:
            # Add to config
            if "scheduled_runs" not in config.get("automation"):
                config.set("automation", "scheduled_runs", [])
            
            config.get("automation", "scheduled_runs").append({
                "time": time_str,
                "script": script,
                "duration": duration,
                "params": params or {}
            })
            
            # Add to scheduler if running
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                schedule.every().day.at(time_str).do(
                    self._run_process, script, duration, params
                )
            
            logger.info(f"Added scheduled run for {script} at {time_str}")
            return True
        except Exception as e:
            logger.error(f"Failed to add scheduled run: {e}")
            return False
    
    def remove_scheduled_run(self, time_str, script):
        """Remove a scheduled run."""
        try:
            # Remove from config
            scheduled_runs = config.get("automation", "scheduled_runs")
            if scheduled_runs:
                config.set("automation", "scheduled_runs", [
                    run for run in scheduled_runs
                    if not (run["time"] == time_str and run.get("script") == script)
                ])
            
            # Remove from scheduler if running
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                schedule.clear()
                
                # Re-add remaining scheduled runs
                for run in config.get("automation", "scheduled_runs"):
                    schedule.every().day.at(run["time"]).do(
                        self._run_process, run.get("script", "main.py"),
                        run.get("duration"), run.get("params")
                    )
            
            logger.info(f"Removed scheduled run for {script} at {time_str}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove scheduled run: {e}")
            return False
    
    def run_now(self, script, duration=None, params=None):
        """Run a script immediately."""
        return self._run_process(script, duration, params)
    
    def terminate_process(self, pid):
        """Terminate a running process."""
        with self.lock:
            if pid in self.running_processes:
                process = self.running_processes[pid]["process"]
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    logger.info(f"Terminated process {pid}")
                del self.running_processes[pid]
                return True
            else:
                logger.warning(f"Process {pid} not found")
                return False
    
    def list_processes(self):
        """List all running processes."""
        with self.lock:
            return {
                pid: {
                    "script": info["script"],
                    "start_time": datetime.datetime.fromtimestamp(info["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": info["duration"],
                    "running_time": int(time.time() - info["start_time"])
                }
                for pid, info in self.running_processes.items()
                if info["process"].poll() is None  # Only include running processes
            }

# Create a singleton instance
scheduler = Scheduler()