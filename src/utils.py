# utils.py

import os

def file_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "outputs")
    
    paths = {
        "PROJECT_ROOT": project_root,
        "DATA_DIR": data_dir,
        "IMG_DIR": os.path.join(data_dir, "images"),
        "OUTPUT_DIR": output_dir,
        "MODEL_DIR": os.path.join(output_dir, "model_weights"),
        "PLOT_DIR": os.path.join(output_dir, "plots"),
    }
    
    # Create directories if they don't exist
    for path in ["OUTPUT_DIR", "MODEL_DIR", "PLOT_DIR"]:
        os.makedirs(paths[path], exist_ok=True)
    
    return paths