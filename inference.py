import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# 1. Setup Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("No GPU found! using CPU")
    device = torch.device("cpu")

# Re-instantiate your specific BitUNet class here
# model = BitUNet(n_channels=3, n_classes=3) 
model.load_state_dict(torch.load("model_path.pth", map_location=device))
model.to(device)
model.eval()

# 2. Define Preprocessing (Must match your joint_augment logic)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define a color map for your 3 classes
# 0: Black (BG), 1: Green (Road), 2: Red (Obstacle)
color_map = {
    0: [0, 0, 0],       
    1: [0, 255, 0],     
    2: [0, 0, 255]      
}

def apply_mask(image, mask):
    """Overlays the segmentation mask onto the original image."""
    mask_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in color_map.items():
        mask_img[mask == cls] = color
    
    # Blend the original frame and the mask (50% transparency)
    return cv2.addWeighted(image, 0.7, mask_img, 0.3, 0)

# 3. Video Processing Loop
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Setup Output Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_result.mp4', fourcc, fps, (width, height))

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.get()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Preprocess & Inference
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        output = model(input_tensor) # Shape: [1, 3, 256, 256]
        
        # Get class predictions
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize mask back to original frame size for overlay
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        # Visual results
        result_frame = apply_mask(frame, mask_resized)
        
        # Write/Show
        out.write(result_frame)
        cv2.imshow('BitUNet Inference', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()