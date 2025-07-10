    import torch

    try:
        model = torch.load('C:/Users/Kaif Ibrahim/Desktop/solinas_downloads/yolov9/runs/train/WRc_water_v9_2.2.0/New folder/best.pt')
        print("File loaded successfully. It is likely not corrupted.")
    except RuntimeError as e:
        if "invalid header or archive is corrupted" in str(e):
            print(f"Error: The .pt file appears to be corrupted: {e}")
        else:
            print(f"An unexpected error occurred: {e}")
    except Exception as e:
        print(f"An error occurred during loading: {e}")