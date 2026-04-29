# import cv2
# import numpy as np
# import torch
# import torchvision
# import torch.nn as nn
# import collections

# # ─────────────────────────────────────────────
# # 1. LOAD MODEL
# # ─────────────────────────────────────────────
# BEST_MODEL_PATH = "best_model_v4.pth"

# class LightweightMalpracticeModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         base_model = torchvision.models.mobilenet_v2(
#             weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
#         )
#         first_conv = base_model.features[0][0]
#         new_conv   = nn.Conv2d(6, first_conv.out_channels,
#                                kernel_size=first_conv.kernel_size,
#                                stride=first_conv.stride,
#                                padding=first_conv.padding,
#                                bias=False)
#         with torch.no_grad():
#             new_conv.weight[:, :3] = first_conv.weight
#             new_conv.weight[:, 3:] = first_conv.weight
#         base_model.features[0][0] = new_conv
#         self.cnn  = base_model.features
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.classifier = nn.Sequential(
#             nn.Linear(1280, 128),
#             nn.ReLU(),
#             nn.Dropout(0.6),
#             nn.Linear(128, 1)
#         )

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         x     = x.view(B * T, C, H, W)
#         feats = self.cnn(x)
#         feats = self.pool(feats).view(B, T, -1)
#         mean_pool = feats.mean(dim=1)
#         max_pool  = feats.max(dim=1).values
#         feats     = 0.5 * mean_pool + 0.5 * max_pool
#         return self.classifier(feats)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model  = LightweightMalpracticeModel().to(device)

# checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
# model.load_state_dict(checkpoint["model_state"])
# model.eval()
# print(f"✅ Model loaded from epoch {checkpoint['epoch']}")

# # ─────────────────────────────────────────────
# # 2. INFERENCE SETTINGS
# # ─────────────────────────────────────────────
# NUM_FRAMES  = 16     # frames per clip (same as training)
# SIZE        = (112, 112)
# THRESHOLD   = 0.4      # raise to 0.6 to reduce false alarms
# SMOOTHING   = 10         # average last N predictions for stability

# frame_buffer    = collections.deque(maxlen=NUM_FRAMES)
# pred_buffer     = collections.deque(maxlen=SMOOTHING)
# current_label   = "Waiting..."
# current_conf    = 0.0
# current_color   = (200, 200, 200)

# def predict_clip(frames):
#     clip = np.stack(frames, axis=0).astype(np.float32)   # (64, 112, 112, 3)
#     clip = np.clip(clip, 0.0, 1.0)
#     clip = np.concatenate([clip, np.zeros_like(clip)], axis=-1)  # (64,112,112,6)
#     clip = np.transpose(clip, (0, 3, 1, 2))              # (64, 6, 112, 112)
#     clip = torch.from_numpy(clip).float().unsqueeze(0).to(device)  # (1,64,6,112,112)

#     with torch.no_grad():
#         output = model(clip)
#         prob   = torch.sigmoid(output).item()

#     return prob

# # ─────────────────────────────────────────────
# # 3. REAL-TIME LOOP
# # ─────────────────────────────────────────────
# cap = cv2.VideoCapture(0)  # 0 = default webcam

# if not cap.isOpened():
#     print("❌ Cannot open webcam")
# else:
#     print("✅ Webcam opened — press Q to quit")

# frame_count = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Failed to grab frame")
#         break
    
#     # 1. Mirror Effect
#     frame = cv2.flip(frame, 1)

#     # 2. Square Center Crop (Fixes Portrait/Landscape mismatch)
#     h, w = frame.shape[:2]
#     min_dim = min(h, w)
#     start_x = (w - min_dim) // 2
#     start_y = (h - min_dim) // 2
    
#     # Slicing: [y_start:y_end, x_start:x_end]
#     cropped_frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

#     # 3. Preprocess for AI
#     resized = cv2.resize(cropped_frame, SIZE) # SIZE is (112, 112)
#     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0
#     frame_buffer.append(rgb)
#     frame_count += 1

#     # Run prediction every 32 frames once buffer is full
#     if len(frame_buffer) == NUM_FRAMES and frame_count % 8 == 0:
#         prob = predict_clip(list(frame_buffer))
#         pred_buffer.append(prob)

#         # Smooth predictions
#         smooth_prob = np.mean(pred_buffer)

#         if smooth_prob >= THRESHOLD:
#             current_label = "⚠ MALPRACTICE"
#             current_color = (0, 0, 255)    # Red
#         else:
#             current_label = "✓ NORMAL"
#             current_color = (0, 200, 0)    # Green
#         current_conf = smooth_prob

#     # ─────────────────────────────────────────────
#     # 4. DISPLAY
#     # ─────────────────────────────────────────────
#     # Status bar background
#     cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (30, 30, 30), -1)

#     # Label + confidence
#     cv2.putText(frame, current_label,
#                 (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
#                 1.0, current_color, 2)
#     cv2.putText(frame, f"Confidence: {current_conf:.2f}",
#                 (frame.shape[1] - 220, 35),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Buffer fill progress bar
#     fill = int((len(frame_buffer) / NUM_FRAMES) * frame.shape[1])
#     cv2.rectangle(frame, (0, frame.shape[0]-8),
#                   (fill, frame.shape[0]), (100, 180, 255), -1)
#     cv2.putText(frame, "Buffer",
#                 (5, frame.shape[0] - 12),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

#     cv2.imshow("Malpractice Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print("👋 Quit")
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import collections
import time
import datetime
import os

# ─────────────────────────────────────────────
# 1. LOAD MODEL (UNCHANGED)
# ─────────────────────────────────────────────
BEST_MODEL_PATH = "best_model_v4.pth"

class LightweightMalpracticeModel(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = torchvision.models.mobilenet_v2(
            weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        first_conv = base_model.features[0][0]
        new_conv = nn.Conv2d(6, first_conv.out_channels,
                             kernel_size=first_conv.kernel_size,
                             stride=first_conv.stride,
                             padding=first_conv.padding,
                             bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = first_conv.weight
            new_conv.weight[:, 3:] = first_conv.weight
        base_model.features[0][0] = new_conv

        self.cnn = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)
        feats = self.pool(feats).view(B, T, -1)
        mean_pool = feats.mean(dim=1)
        max_pool = feats.max(dim=1).values
        feats = 0.5 * mean_pool + 0.5 * max_pool
        return self.classifier(feats)

device = torch.device("cpu")
model = LightweightMalpracticeModel().to(device)

checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("✅ Model loaded")

# ─────────────────────────────────────────────
# 2. SETTINGS (IMPORTANT)
# ─────────────────────────────────────────────
NUM_FRAMES = 16
SIZE = (112, 112)
SMOOTHING = 10

START_THRESHOLD = 0.6
MIN_DURATION = 8      # seconds required to confirm malpractice
COOLDOWN = 60

# FPS = cap.get(cv2.CAP_PROP_FPS)
# if FPS == 0:
FPS = 20
cap = cv2.VideoCapture(0)
PRE_SECONDS = 5
POST_SECONDS = 5
BUFFER_SIZE = PRE_SECONDS * FPS

# Buffers
frame_buffer = collections.deque(maxlen=NUM_FRAMES)
pred_buffer = collections.deque(maxlen=SMOOTHING)
video_buffer = collections.deque(maxlen=BUFFER_SIZE)

# State
last_record_time = 0
recording = False
record_end_time = 0
video_writer = None

suspicious_time = 0
last_time = time.time()
DECAY = 1.5   # how fast it reduces when normal

current_label = "Waiting..."
current_conf = 0.0
current_color = (200, 200, 200)

os.makedirs("evidence/videos", exist_ok=True)

# ─────────────────────────────────────────────
# 3. PREDICTION
# ─────────────────────────────────────────────
def predict_clip(frames):
    clip = np.stack(frames, axis=0).astype(np.float32)
    clip = np.clip(clip, 0.0, 1.0)
    clip = np.concatenate([clip, np.zeros_like(clip)], axis=-1)
    clip = np.transpose(clip, (0, 3, 1, 2))
    clip = torch.from_numpy(clip).float().unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(clip)
        prob = torch.sigmoid(output).item()

    return prob

# ─────────────────────────────────────────────
# 4. CAMERA LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera error")
    exit()

print("✅ Running... Press Q to quit")

start_time = time.time()
WARMUP = 3   # seconds
# Alert sound file (put alert.wav in same folder)
ALERT_SOUND = "alert.wav"

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if time.time() - start_time < WARMUP:
        continue
    frame = cv2.flip(frame, 1)

    # Save to pre-buffer
    video_buffer.append(frame.copy())

    # Crop center square
    h, w = frame.shape[:2]
    m = min(h, w)
    x = (w - m) // 2
    y = (h - m) // 2
    cropped = frame[y:y+m, x:x+m]

    # Preprocess
    resized = cv2.resize(cropped, SIZE)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) / 255.0

    frame_buffer.append(rgb)
    frame_count += 1

    # Inference (reduced frequency)
    if len(frame_buffer) == NUM_FRAMES and frame_count % 12 == 0:
        prob = predict_clip(list(frame_buffer))
        pred_buffer.append(prob)
        smooth_prob = np.mean(pred_buffer)
        current_conf = smooth_prob

        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if smooth_prob >= START_THRESHOLD:
            suspicious_time += dt
        else:
            suspicious_time -= dt * DECAY

        # clamp
        suspicious_time = max(0, suspicious_time)

        alert = suspicious_time >= MIN_DURATION

        # Trigger recording
        if alert and not recording and (current_time - last_record_time > COOLDOWN):

            last_record_time = current_time
            current_label = "⚠ MALPRACTICE"
            current_color = (0, 0, 255)

            import os

            if os.name == "nt":   # Windows
                import winsound
                if os.path.exists(ALERT_SOUND):
                    winsound.PlaySound(ALERT_SOUND, winsound.SND_ASYNC)
            else:   # Linux / Raspberry Pi
                if os.path.exists(ALERT_SOUND):
                    os.system(f"aplay {ALERT_SOUND} &")

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                f"evidence/videos/{ts}.avi",
                fourcc, FPS,
                (frame.shape[1], frame.shape[0])
            )

            # Write previous frames (5 sec before)
            for bf in video_buffer:
                video_writer.write(bf)

            recording = True
            record_end_time = current_time + POST_SECONDS
        else:
            current_label = "✓ NORMAL"
            current_color = (0, 200, 0)

        # Continue recording
        if recording:
            video_writer.write(frame)

            if time.time() > record_end_time:
                recording = False
                video_writer.release()
                print("📁 Video saved")

    # Timestamp
    ts_display = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts_display, (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # UI
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (30,30,30), -1)
    cv2.putText(frame, current_label, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, current_color, 2)
    cv2.putText(frame, f"Conf: {current_conf:.2f}",
                (frame.shape[1]-200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Malpractice Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()