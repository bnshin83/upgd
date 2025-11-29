import json
import os
import numpy as np


def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON serializable Python types"""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # For PyTorch tensors
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class Logger:
    """
    This class is responsible for logging the training process.
    Log file will be saved in the log directory with JSON format.
    """

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.log_path = None
        self.log_dir_path = None

    def initialize_log_path(self, **kwargs):
        """Initialize the log directory and file path early"""
        file_name = ''
        for key, value in kwargs["optimizer_hps"].items():
            file_name += f'{key}_{value}_'
        
        # Add n_samples to the folder name for differentiation
        if 'n_samples' in kwargs:
            file_name += f'n_samples_{kwargs["n_samples"]}_'
            
        file_name = file_name[:-1]
        
        self.log_dir_path = f"{self.log_dir}/{kwargs['task']}/{kwargs['learner']}/{kwargs['network']}/{file_name}/"
        self.log_path = f"{self.log_dir_path}/{kwargs['seed']}.json"
        
        # Create directory early
        if not os.path.exists(self.log_dir_path):
            os.makedirs(self.log_dir_path)
            print(f"Created log directory: {self.log_dir_path}")

    def log(self, **kwargs):
        # Initialize path if not already done
        if self.log_path is None:
            self.initialize_log_path(**kwargs)
        
        # Convert all data to JSON serializable types
        serializable_data = convert_to_serializable(kwargs)
        json_object = json.dumps(serializable_data, indent=4)

        with open(self.log_path, "w") as outfile:
            outfile.write(json_object)

    def log_incremental(self, data_dict, **meta_kwargs):
        """Log incremental data without overwriting, useful for partial updates"""
        if self.log_path is None:
            self.initialize_log_path(**meta_kwargs)
        
        # Try to load existing data, or start with empty dict
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, "r") as infile:
                    existing_data = json.load(infile)
            else:
                existing_data = {}
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = {}
        
        # Update with new data
        existing_data.update(convert_to_serializable(data_dict))
        
        # Write back to file
        json_object = json.dumps(existing_data, indent=4)
        with open(self.log_path, "w") as outfile:
            outfile.write(json_object)

