import json
import os
import os.path as op
from datetime import datetime

class PaperDataManager:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.file_path = op.join(results_dir, 'paper_dataset.json')
        self.data = self._load_data()

    def _load_data(self):
        if op.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def add_result(self, analysis_type, subject, metric_name, value, metadata=None):
        """
        Adds a data point to the dataset.
        
        Args:
            analysis_type (str): e.g., "slow_wave_erp", "alpha_beta_tfr", "profiling"
            subject (str): Subject ID or "Group"
            metric_name (str): e.g., "p_value", "peak_amplitude", "classification"
            value (any): The actual data value (float, string, list)
            metadata (dict, optional): Extra info like {time_window: "0.5-1.0s", region: "frontal"}
        """
        if analysis_type not in self.data:
            self.data[analysis_type] = {}
        
        if subject not in self.data[analysis_type]:
            self.data[analysis_type][subject] = {}
            
        entry = {
            "value": value,
            "updated_at": datetime.now().isoformat()
        }
        if metadata:
            entry.update(metadata)
            
        self.data[analysis_type][subject][metric_name] = entry
        self.save()

    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=4, sort_keys=True)
        print(f"[PaperData] Saved {self.file_path}")

# Helper to get instance easily in scripts
def get_data_manager(config):
    return PaperDataManager(config['paths']['results_dir'])
