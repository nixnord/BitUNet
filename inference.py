import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from time import perf_counter
from queue import Queue
import threading

class InferenceBitConv2d(nn.Module):
    """Loads INT8 weights and FP32 scales for memory-efficient inference."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.padding = padding
        self.register_buffer('weight', torch.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0, dtype=torch.float32))

        if bias:
            self.register_buffer('bias', torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.bias = None

    def forward(self, x):
        # 1. Quantize activations
        x_scale = 127.0 / x.abs().max(dim=1, keepdim=True)[0].clamp_(min=1e-5)
        x_quant = torch.round(x * x_scale).clamp_(-128, 127) / x_scale

        # 2. Dequantize weights on the fly using the exported scale
        w_fp32 = self.weight.to(x.dtype) * self.weight_scale

        return F.conv2d(x_quant, w_fp32, self.bias, padding=self.padding)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_bit=True):
        super().__init__()
        Conv = InferenceBitConv2d if use_bit else nn.Conv2d
        self.block = nn.Sequential(
            Conv(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            Conv(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)



class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.block(x)



class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        dH = skip.size(2) - x.size(2)
        dW = skip.size(3) - x.size(3)
        x = F.pad(x, [dW//2, dW-dW//2, dH//2, dH-dH//2])
        return self.conv(torch.cat([skip, x], dim=1))



class BitUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, base_channels=72):
        super().__init__()
        c = base_channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.bridge = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c*8, c*16, use_bit=True))
        self.up3 = Up(c*16 + c*8, c*8)
        self.up2 = Up(c*8 + c*4, c*4)
        self.up1 = Up(c*4 + c*2, c*2)
        self.up0 = Up(c*2 + c, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b  = self.bridge(s4)
        d3 = self.up3(b, s4)
        d2 = self.up2(d3, s3)
        d1 = self.up1(d2, s2)
        d0 = self.up0(d1, s1)
        return self.head(d0)


def joint_augment(image, mask, img_size=(256,256), is_train=True):
    image = TF.resize(image, img_size, TF.InterpolationMode.BILINEAR)
    mask  = TF.resize(mask,  img_size, TF.InterpolationMode.NEAREST)

    # Force mask back to L mode after resize (torchvision can silently convert it)
    if mask.mode != "L":
        mask = mask.convert("L")

    if is_train:
        if random.random() > 0.5:
            image = TF.hflip(image); mask = TF.hflip(mask)

        crop_h = int(img_size[0] * 0.9)
        crop_w = int(img_size[1] * 0.9)
        i,j,h,w = T.RandomCrop.get_params(image, (crop_h, crop_w))
        image = TF.resized_crop(image, i,j,h,w, img_size, TF.InterpolationMode.BILINEAR)
        mask  = TF.resized_crop(mask,  i,j,h,w, img_size, TF.InterpolationMode.NEAREST)

        # Force again after crop (same issue)
        if mask.mode != "L":
            mask = mask.convert("L")

        image = TF.adjust_brightness(image, 1 + random.uniform(-0.2, 0.2))
        image = TF.adjust_contrast(image,   1 + random.uniform(-0.2, 0.2))

    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    # Convert mask safely — always 2D
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    mask = torch.from_numpy(mask_np).long()

    return image, mask


MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def unnormalize(tensor):
    """Convert normalised image tensor [3,H,W] back to uint8 [H,W,3]."""
    img = tensor.permute(1,2,0).numpy()   # [H,W,3]
    img = img * STD + MEAN                # undo normalisation
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


INPUT_VIDEO_PATH = "test_videos/test_video.mp4"
OUTPUT_VIDEO_PATH = "segmented_output.avi"
MODEL_WEIGHTS = "bitnet_ternary_exported.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLASS_COLORS = {
    0: np.array([0, 0, 0]),       # Background
    1: np.array([0, 255, 0]),     # Road
    2: np.array([255, 0, 0])      # Obstacle
}



def load_ternary_model(model_path, device):
    """Loads the BitUNet model strictly mapping INT8 weights."""
    if not os.path.exists(model_path):
        print(f"Error: Weights file '{model_path}' not found.")
        return None

    print("Loading INT8 model architecture...")
    model = BitUNet(in_channels=3, num_classes=3, base_channels=72).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint

        # Load the weights. strict=True ensures our inference layer matches perfectly.
        model.load_state_dict(state_dict, strict=True)
        print("Ternary model loaded successfully!")

    except Exception as e:
        print(f"Could not load weights: {e}")
        return None

    model.eval()
    return model

def predict_video(model, input_path, output_path, device, batch_size=14):
    """Runs batch inference on a video to maximize GPU throughput."""
    if not os.path.exists(input_path):
        print(f"Error: Video file '{input_path}' not found.")
        return

    # 1. Initialize Video Capture & Writer
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_width, out_height = 256, 256

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"Processing video: {total_frames} frames at {fps:.1f} FPS (Batch Size: {batch_size})...")
    frame_count = 0
    start_time = perf_counter()
    inference_times = []
    # Buffers to hold our batches
    batch_tensors = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            # If we successfully read a frame, preprocess it
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                raw_img = Image.fromarray(frame_rgb)
                dummy_mask = Image.new("L", raw_img.size)

                img_t, _ = joint_augment(raw_img, dummy_mask, img_size=(256, 256), is_train=False)
                batch_tensors.append(img_t)

            # If our batch is full, OR the video ended but we still have frames in the buffer
            if len(batch_tensors) == batch_size or (not ret and len(batch_tensors) > 0):
                print(len(batch_tensors))
                # 1. Stack the list of tensors into a single batch: [Batch, 3, 256, 256]
                img_t_batch = torch.stack(batch_tensors).to(device)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = perf_counter()
                # 2. Parallel Inference on the GPU
                with torch.no_grad():
                    logits = model(img_t_batch)
                    # Get predictions for the whole batch at once: [Batch, 256, 256]
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = perf_counter()
                duration_ms = (end - start) * 1000
                inference_times.append(duration_ms)
                # 3. CPU Post-processing & Writing (Loop through the processed batch)
                for i in range(len(batch_tensors)):
                    # Grab the individual tensor and prediction from the batch
                    single_img_t = batch_tensors[i].cpu()
                    pred = preds[i]

                    # Visualization & Blending
                    img_display = unnormalize(single_img_t)
                    img_display_float = img_display.astype(np.float32)
                    if img_display_float.max() > 2.0:
                        img_display_float /= 255.0

                    color_mask = np.zeros_like(img_display_float)
                    for cls_idx, color in CLASS_COLORS.items():
                        color_mask[pred == cls_idx] = np.array(color) / 255.0

                    alpha = 0.4
                    overlay_img = (1 - alpha) * img_display_float + alpha * color_mask
                    overlay_img = np.clip(overlay_img, 0.0, 1.0)

                    # Convert to OpenCV format
                    overlay_uint8 = (overlay_img * 255).astype(np.uint8)
                    overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)

                    # Write frame to output video
                    out.write(overlay_bgr)

                    # Progress tracker
                    frame_count += 1
                    if frame_count % 100 == 0:
                        current_avg = sum(inference_times) / len(inference_times)
                        print(f"Processed {frame_count}/{total_frames} frames... | Avg latency: {current_avg:.2f} ms")

                # Clear the buffer for the next batch
                batch_tensors = []

            # Break the while loop if the video has ended
            if not ret:
                break

    except KeyboardInterrupt:
        print("\nInference stopped early by user. Finalizing the video...")

    finally:
        cap.release()
        out.release()
        end_time = perf_counter()
        total_time = round(end_time - start_time, 2)
        achieved_fps = round(frame_count / total_time, 2) if total_time > 0 else 0
        avg_time = sum(inference_times) / len(inference_times)
        print(f"Video safely finalized and saved to '{output_path}'")
        print(f"Total time taken: {total_time} seconds")
        print(f"Average Inference Speed: {achieved_fps} FPS")
        print(f"Average Inference Time: {avg_time:.2f} ms per frame")

def predict_video_async(model, input_path, output_path, device, batch_size=14):
    """Runs multithreaded batch inference on a video to maximize GPU throughput."""
    if not os.path.exists(input_path):
        print(f"Error: Video file '{input_path}' not found.")
        return

    # 1. Initialize Video Capture & Writer
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_width, out_height = 256, 256

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"Processing video: {total_frames} frames at {fps:.1f} native FPS (Batch Size: {batch_size})...")
    
    # 2. Setup Asynchronous Queues
    # Maxsize prevents RAM explosions if the reader is faster than the GPU
    input_queue = Queue(maxsize=batch_size * 4) 
    output_queue = Queue(maxsize=batch_size * 4)

    # Tracking variables (Use lists so threads can modify them)
    inference_times = []
    frames_processed = [0] 

    # --- THREAD 1: The Reader ---
    def reader_worker():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_img = Image.fromarray(frame_rgb)
            dummy_mask = Image.new("L", raw_img.size)

            img_t, _ = joint_augment(raw_img, dummy_mask, img_size=(256, 256), is_train=False)
            input_queue.put(img_t) # Push to GPU waiting line
            
        input_queue.put(None) # Poison pill to tell the GPU the video is done

    # --- THREAD 2: The Writer ---
    def writer_worker():
        while True:
            item = output_queue.get()
            if item is None: # Poison pill received, stop writing
                break
                
            batch_tensors, preds = item
            
            # CPU Post-processing & Writing 
            for i in range(len(batch_tensors)):
                single_img_t = batch_tensors[i].cpu()
                pred = preds[i]

                img_display = unnormalize(single_img_t)
                img_display_float = img_display.astype(np.float32)
                if img_display_float.max() > 2.0:
                    img_display_float /= 255.0

                color_mask = np.zeros_like(img_display_float)
                for cls_idx, color in CLASS_COLORS.items():
                    color_mask[pred == cls_idx] = np.array(color) / 255.0

                alpha = 0.4
                overlay_img = (1 - alpha) * img_display_float + alpha * color_mask
                overlay_img = np.clip(overlay_img, 0.0, 1.0)

                overlay_uint8 = (overlay_img * 255).astype(np.uint8)
                overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)

                out.write(overlay_bgr)
                
                frames_processed[0] += 1
                if frames_processed[0] % 100 == 0:
                    print(f"Written {frames_processed[0]}/{total_frames} frames to disk...")

    # 3. Start the helper threads
    reader_thread = threading.Thread(target=reader_worker)
    writer_thread = threading.Thread(target=writer_worker)
    reader_thread.start()
    writer_thread.start()

    # --- MAIN THREAD: The GPU Engine ---
    start_time_total = perf_counter()
    batch_tensors = []

    try:
        while True:
            # Get preprocessed frame from reader
            img_t = input_queue.get()
            
            if img_t is not None:
                batch_tensors.append(img_t)

            # If batch is full, or we received the EOF signal but have leftover frames
            if len(batch_tensors) == batch_size or (img_t is None and len(batch_tensors) > 0):
                img_t_batch = torch.stack(batch_tensors).to(device)
                
                if device.type == 'cuda': torch.cuda.synchronize()
                start_inf = perf_counter()
                
                with torch.no_grad():
                    logits = model(img_t_batch)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    
                if device.type == 'cuda': torch.cuda.synchronize()
                end_inf = perf_counter()
                
                # Note: We divide by len(batch_tensors) to get the true *per frame* latency
                inference_times.append(((end_inf - start_inf) * 1000) / len(batch_tensors))
                
                # Push the finished batch to the writer thread
                output_queue.put((batch_tensors, preds))
                batch_tensors = []

            if img_t is None:
                output_queue.put(None) # Tell writer we are done
                break

    except KeyboardInterrupt:
        print("\nInference stopped early by user. Emptying queues and finalizing...")
        output_queue.put(None)

    finally:
        # 4. Clean up and wait for threads to finish saving the file
        reader_thread.join()
        writer_thread.join()
        cap.release()
        out.release()
        
        end_time_total = perf_counter()
        total_time = round(end_time_total - start_time_total, 2)
        achieved_fps = round(frames_processed[0] / total_time, 2) if total_time > 0 else 0
        avg_time = sum(inference_times) / len(inference_times) if inference_times else 0
        
        print(f"\n--- ASYNC PERFORMANCE SUMMARY ---")
        print(f"Video safely finalized and saved to '{output_path}'")
        print(f"Total time taken: {total_time} seconds")
        print(f"End-to-End Processing Speed: {achieved_fps} FPS")
        print(f"Pure GPU Inference Latency: {avg_time:.2f} ms per frame")


'''
def predict_video(model, input_path, output_path, device):
    """Runs inference frame-by-frame on a video and saves the segmented output safely."""
    if not os.path.exists(input_path):
        print(f"Error: Video file '{input_path}' not found.")
        return

    # 1. Initialize Video Capture
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width, out_height = 256, 256

    # FIX 1: Change codec to 'avc1' (H.264) which is much better supported on Linux
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"Processing video: {total_frames} frames at {fps:.1f} FPS...")
    frame_count = 0
    start_time = perf_counter()
    # FIX 2: Use try/finally to ensure the video always saves, even if you interrupt it
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_img = Image.fromarray(frame_rgb)
            dummy_mask = Image.new("L", raw_img.size)

            img_t, _ = joint_augment(raw_img, dummy_mask, img_size=(256, 256), is_train=False)
            img_t_batch = img_t.unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                logits = model(img_t_batch)
                pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            # Visualization & Blending
            img_display = unnormalize(img_t.squeeze(0).cpu())
            img_display_float = img_display.astype(np.float32)
            if img_display_float.max() > 2.0:
                img_display_float /= 255.0

            color_mask = np.zeros_like(img_display_float)
            for cls_idx, color in CLASS_COLORS.items():
                color_mask[pred == cls_idx] = np.array(color) / 255.0

            alpha = 0.4
            overlay_img = (1 - alpha) * img_display_float + alpha * color_mask
            overlay_img = np.clip(overlay_img, 0.0, 1.0)

            # Convert to OpenCV format
            overlay_uint8 = (overlay_img * 255).astype(np.uint8)
            overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)

            # Write frame to output video
            out.write(overlay_bgr)

            # Progress tracker
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

    except KeyboardInterrupt:
        # If you press Ctrl+C, it catches it here gracefully
        print("\nInference stopped early by user. Finalizing the video...")

    finally:
        # This guarantees the video file is closed and saved properly no matter what
        cap.release()
        out.release()
        end_time = perf_counter()
        print(f"Video safely finalized and saved to '{output_path}'")
        print(f"time taken: {round(end_time - start_time, 2)}")
'''

if __name__ == "__main__":
    loaded_model = load_ternary_model(MODEL_WEIGHTS, DEVICE)
    if loaded_model is not None:
        predict_video_async(loaded_model, INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, DEVICE)
