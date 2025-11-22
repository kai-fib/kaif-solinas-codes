from roboflow import Roboflow

rf = Roboflow(api_key="M9Yx3y22OHZLl0WQHm9A")

# Workspace ID from URL
workspace = rf.workspace("datasetiv")

# Deploy YOLOv9 model to Roboflow
workspace.deploy_model(
    model_type="yolov9",  # Your model type
    model_path=r"D:\yolov9\runs\train\wrc\weights",
    project_ids=["train07-sgzgf"],  # Project ID from URL
    model_name="wrc-sewer-v9",      # Custom name for model
    filename="best.pt"              # Name of weights file
)
