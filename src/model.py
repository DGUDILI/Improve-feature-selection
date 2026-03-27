import subprocess
import os

class Model:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(self):
        print("Running StackDILI Fixed pipeline...")

        scripts = [
            os.path.join(self.project_root, "src", "preprocessing", "make_clean_data.py"),
            os.path.join(self.project_root, "src", "models", "stacking", "stacking.py"),
            os.path.join(self.project_root, "src", "models", "evaluate.py"),
        ]

        for script in scripts:
            print(f"Executing {script}...")
            subprocess.run(["python", script], text=True, check=True)

        result_path = os.path.join(self.project_root, "src", "models", "result.txt")
        if os.path.exists(result_path):
            with open(result_path) as f:
                auc = f.read()
            print("StackDILI Fixed AUC:", auc)
        else:
            print("result.txt not found")

    def predict(self, _):
        return None
