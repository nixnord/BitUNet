import os
import cv2
import torch
import numpy as np
from PIL import Image
from time import perf_counter
from queue import Queue
import threading
from inference import (
    joint_augment,
    unnormalize,
    INPUT_VIDEO_PATH,
    OUTPUT_VIDEO_PATH,
    MODEL_WEIGHTS,
    DEVICE,
    CLASS_COLORS,
    load_ternary_model
)
class Kalman1D:
    def __init__(self):
        self.x = 0; self.p = 1; self.q = 0.01; self.r = 5
    def update(self, z):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (z - self.x)
        self.p *= (1 - k)
        return self.x

def extract_points(mask, kalman_filters):
    H, W = mask.shape
    points = []
    idx = 0
    for y in range(int(H * 0.7), H, 5):
        xs = np.where(mask[y] == 1)[0]
        if len(xs) > 20:
            x = int(np.median(xs))
            x = int(kalman_filters[idx].update(x))
            points.append((x, y))
            idx += 1
    return points

def fit_curve(points):
    if len(points) < 6: return None
    pts = np.array(points)
    return np.polyfit(pts[:, 1], pts[:, 0], 2)

def draw_trajectory(frame, coeffs):
    if coeffs is None: return frame
    h, w = frame.shape[:2]

    ys = np.linspace(int(h * 0.7), h, 40)
    xs = coeffs[0]*ys**2 + coeffs[1]*ys + coeffs[2]
    pts = [(int(np.clip(x, 0, w-1)), int(y)) for x, y in zip(xs, ys)]
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (255, 0, 0), 3)

    y_bottom = h - 10
    slope = 2 * coeffs[0] * y_bottom + coeffs[1]

    if slope > 0.3:
        text, color = "RIGHT", (0, 255, 255)
    elif slope < -0.3:
        text, color = "LEFT", (0, 255, 255)
    else:
        text, color = "STRAIGHT", (0, 255, 0)

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def predict_path_async_fp16(model, input_path, output_path, device, batch_size=30):
    """Runs FP16 multithreaded batch inference with path planning at 256x256 resolution."""
    if not os.path.exists(input_path):
        print(f"Error: Video file '{input_path}' not found.")
        return

    print("Converting floating-point layers to FP16 (Half-Precision)...")
    model = model.half()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_width, out_height = 256, 256

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"Processing video: {total_frames} frames at {fps:.1f} native FPS (Batch Size: {batch_size})...")
    print(f"Output Resolution: {out_width}x{out_height}")

    input_queue = Queue(maxsize=batch_size * 4)
    output_queue = Queue(maxsize=batch_size * 4)

    inference_times = []
    frames_processed = [0]
    start_time = [perf_counter()]

    kalman_filters = [Kalman1D() for _ in range(50)]

    def reader_worker():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_img = Image.fromarray(frame_rgb)
            dummy_mask = Image.new("L", raw_img.size)

            img_t, _ = joint_augment(raw_img, dummy_mask, img_size=(256, 256), is_train=False)
            input_queue.put((img_t, frame))

        input_queue.put(None)

    def writer_worker():
        while True:
            item = output_queue.get()
            if item is None: break

            batch_frames, preds = item

            for i in range(len(batch_frames)):
                frame = batch_frames[i]
                pred = preds[i]
                frame_256 = cv2.resize(frame, (256, 256))

                color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
                for cls_idx, color in CLASS_COLORS.items():
                    color_mask[pred == cls_idx] = color
                color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(frame_256, 0.7, color_mask_bgr, 0.3, 0)

                points = extract_points(pred, kalman_filters)
                coeffs = fit_curve(points)
                if coeffs is not None:
                    ys = np.linspace(int(256 * 0.7), 256, 40)
                    xs = coeffs[0]*ys**2 + coeffs[1]*ys + coeffs[2]
                    pts = [(int(np.clip(x, 0, 255)), int(y)) for x, y in zip(xs, ys)]

                    for j in range(1, len(pts)):
                        cv2.line(overlay, pts[j-1], pts[j], (255, 0, 0), 2)

                    y_bottom = 256 - 10
                    slope = 2 * coeffs[0] * y_bottom + coeffs[1]
                    if slope > 0.3:
                        text, color = "RIGHT", (0, 255, 255)
                    elif slope < -0.3:
                        text, color = "LEFT", (0, 255, 255)
                    else:
                        text, color = "STRAIGHT", (0, 255, 0)

                    cv2.putText(overlay, text, (15, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                current_time = perf_counter()
                elapsed = current_time - start_time[0]
                frames_processed[0] += 1
                current_fps = frames_processed[0] / elapsed if elapsed > 0 else 0
                cv2.putText(overlay, f"FPS: {current_fps:.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                out.write(overlay)

                if frames_processed[0] % 100 == 0 or frames_processed[0] == total_frames:
                    print(f"Written {frames_processed[0]}/{total_frames} frames to disk...")

    reader_thread = threading.Thread(target=reader_worker)
    writer_thread = threading.Thread(target=writer_worker)
    reader_thread.start()
    writer_thread.start()

    batch_tensors = []
    batch_frames = []

    try:
        while True:
            item = input_queue.get()

            if item is not None:
                img_t, frame = item
                batch_tensors.append(img_t)
                batch_frames.append(frame)

            if len(batch_tensors) == batch_size or (item is None and len(batch_tensors) > 0):
                img_t_batch = torch.stack(batch_tensors).to(device).half()

                if device.type == 'cuda': torch.cuda.synchronize()
                start_inf = perf_counter()

                with torch.no_grad():
                    logits = model(img_t_batch)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()

                if device.type == 'cuda': torch.cuda.synchronize()
                end_inf = perf_counter()

                inference_times.append(((end_inf - start_inf) * 1000) / len(batch_tensors))

                output_queue.put((batch_frames, preds))
                batch_tensors = []
                batch_frames = []

            if item is None:
                output_queue.put(None)
                break

    except KeyboardInterrupt:
        print("\nInference stopped early by user. Emptying queues and finalizing...")
        output_queue.put(None)

    finally:
        reader_thread.join()
        writer_thread.join()
        cap.release()
        out.release()

        end_time_total = perf_counter()
        total_time = round(end_time_total - start_time[0], 2)
        achieved_fps = round(frames_processed[0] / total_time, 2) if total_time > 0 else 0
        avg_time = sum(inference_times) / len(inference_times) if inference_times else 0

        print(f"\n--- FP16 ASYNC PERFORMANCE SUMMARY ---")
        print(f"Total time taken: {total_time} seconds")
        print(f"End-to-End Processing Speed: {achieved_fps} FPS")
        print(f"Pure GPU Inference Latency: {avg_time:.2f} ms per frame")

if __name__ == "__main__":
    loaded_model = load_ternary_model(MODEL_WEIGHTS, DEVICE)
    if loaded_model is not None:
        predict_path_async_fp16(loaded_model, INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, DEVICE)