import cv2
import mediapipe as mp
import numpy as np
import time
import warnings
from collections import deque
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

# Suppress protobuf deprecation warning caused by mediapipe
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
# Suppress FastICA convergence warning
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn.decomposition")

# MediaPipe Initialization
mp_face_mesh = mp.solutions.face_mesh

# --- Configurations ---
EAR_THRESHOLD = 0.20
HEAD_YAW_THRESHOLD = 0.20  # Ratio difference from 0.5 (i.e., <0.30 or >0.70)
IRIS_YAW_THRESHOLD = 0.006  # Lowered: iris moves less than head. Difference from 0.5

# --- Landmark Indices ---
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
FACE_SILHOUETTE = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

# --- Helper Functions ---
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    p = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h]) for i in eye_indices]
    ear = (distance(p[1], p[5]) + distance(p[2], p[4])) / (2.0 * distance(p[0], p[3]))
    return ear

def head_yaw_ratio(landmarks, img_w, img_h):
    nose = np.array([landmarks[1].x * img_w, landmarks[1].y * img_h])
    left_cheek = np.array([landmarks[234].x * img_w, landmarks[234].y * img_h])
    right_cheek = np.array([landmarks[454].x * img_w, landmarks[454].y * img_h])
    
    dist_left = distance(nose, left_cheek)
    dist_right = distance(nose, right_cheek)
    
    if dist_left + dist_right == 0:
        return 0.5
    return dist_left / (dist_left + dist_right)

def eye_yaw_ratio(landmarks, left_corner, right_corner, iris_center, img_w, img_h):
    # Use raw x coordinates to estimate horizontal gaze position
    p_lc_x = landmarks[left_corner].x * img_w
    p_rc_x = landmarks[right_corner].x * img_w
    p_ir_x = landmarks[iris_center].x * img_w
    
    # Calculate absolute horizontal distances
    dist_left = abs(p_ir_x - p_lc_x)
    dist_right = abs(p_ir_x - p_rc_x)
    
    if dist_left + dist_right == 0:
        return 0.5
    return dist_left / (dist_left + dist_right)

def get_roi_mean_rgb(image, landmarks, roi_indices, img_w, img_h):
    pts = np.array([(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in roi_indices], dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    mean_val = cv2.mean(image, mask=mask)
    return mean_val[2], mean_val[1], mean_val[0]  # R, G, B

def estimate_hr_rPPG(buffer_r, buffer_g, buffer_b, fps):
    if len(buffer_g) < 100:
        return 0
    
    fs = fps if fps > 0 else 30
    lowcut = 0.75  # 45 BPM
    highcut = 2.75 # 165 BPM
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if low <= 0 or high >= 1:
        return 0
        
    b, a = butter(3, [low, high], btype='band')
    
    try:
        g_filt = filtfilt(b, a, buffer_g)
        r_filt = filtfilt(b, a, buffer_r)
        b_filt = filtfilt(b, a, buffer_b)
        
        X = np.vstack([r_filt, g_filt, b_filt]).T
        
        # Apply FastICA
        ica = FastICA(n_components=3, max_iter=1000, random_state=42)
        S_ = ica.fit_transform(X)
        
        best_hr = 0
        max_power = -1
        
        # Analyze each component in frequency domain
        for i in range(3):
            comp = S_[:, i]
            n = len(comp)
            fft_res = np.fft.rfft(comp)
            freqs = np.fft.rfftfreq(n, d=1.0/fs)
            power = np.abs(fft_res)**2
            
            valid_idx = np.where((freqs >= lowcut) & (freqs <= highcut))[0]
            if len(valid_idx) > 0:
                peak_idx = valid_idx[np.argmax(power[valid_idx])]
                peak_freq = freqs[peak_idx]
                peak_power = power[peak_idx]
                
                if peak_power > max_power:
                    max_power = peak_power
                    best_hr = peak_freq * 60
        return best_hr
    except Exception as e:
        return 0

def calc_cumulative_time(history):
    if len(history) < 2: return 0
    total = 0
    for i in range(1, len(history)):
        if history[i-1][1]:
            total += history[i][0] - history[i-1][0]
    return total

def main():
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    
    # State histories
    head_away_history = deque()
    eyes_away_history = deque()
    
    # Timers
    last_head_forward_time = time.time()
    last_eyes_forward_time = time.time()
    last_eyes_open_time = time.time()
    
    head_forward_start = None
    eyes_forward_start = None
    eyes_open_start = None

    # Persistent States
    state_long_owl = False
    state_short_owl = False
    state_long_lizard = False
    state_short_lizard = False
    state_microsleep = False
    state_sleep = False
    
    # rPPG Buffers
    r_buffer = deque(maxlen=300)
    g_buffer = deque(maxlen=300)
    b_buffer = deque(maxlen=300)
    time_buffer = deque(maxlen=300)
    current_hr = 0
    last_hr_update = time.time()

    prev_frame_time = time.time()
    
    print("DMS Initialized. Press ESC to exit.")

    # Exponential Moving Average (EMA) state
    ema_h_yaw = 0.5
    ema_eye_yaw = 0.5
    ema_ear = 0.3
    
    # Dynamic baseline for eyes (auto-calibration)
    baseline_eye_yaw = 0.5

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        curr_time = time.time()
        fps = 1 / (curr_time - prev_frame_time) if curr_time - prev_frame_time > 0 else 30
        prev_frame_time = curr_time

        image = cv2.flip(image, 1) # Selfie view
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = face_mesh.process(img_rgb)
        
        img_h, img_w, img_c = image.shape
        
        is_face_detected = False
        is_head_away = False
        is_eyes_away = False
        is_eyes_closed = False

        if results.multi_face_landmarks:
            is_face_detected = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # 1. Extraction
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE, img_w, img_h)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE, img_w, img_h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            h_yaw = head_yaw_ratio(face_landmarks.landmark, img_w, img_h)
            
            l_eye_yaw = eye_yaw_ratio(face_landmarks.landmark, 33, 133, 473, img_w, img_h)
            r_eye_yaw = eye_yaw_ratio(face_landmarks.landmark, 362, 263, 468, img_w, img_h)
            avg_eye_yaw = (l_eye_yaw + r_eye_yaw) / 2.0
            
            r, g, b_val = get_roi_mean_rgb(image, face_landmarks.landmark, FACE_SILHOUETTE, img_w, img_h)
            r_buffer.append(r)
            g_buffer.append(g)
            b_buffer.append(b_val)
            time_buffer.append(curr_time)

            # Smooth the signals using EMA to prevent jitter from resetting the timers
            alpha = 0.15
            ema_ear = alpha * avg_ear + (1 - alpha) * ema_ear
            ema_h_yaw = alpha * h_yaw + (1 - alpha) * ema_h_yaw
            ema_eye_yaw = alpha * avg_eye_yaw + (1 - alpha) * ema_eye_yaw
            
            # Very slow EMA to establish the "neutral" resting position of the eyes
            baseline_eye_yaw = 0.002 * ema_eye_yaw + 0.998 * baseline_eye_yaw

            # 2. Condition Evaluation
            is_head_away = abs(ema_h_yaw - 0.5) > HEAD_YAW_THRESHOLD
            # Use dynamic baseline instead of hardcoded 0.5
            is_eyes_away = abs(ema_eye_yaw - baseline_eye_yaw) > IRIS_YAW_THRESHOLD and not is_head_away
            is_eyes_closed = ema_ear < EAR_THRESHOLD
            
            # Print debug info on screen
            cv2.putText(image, f"Eye Yaw: {ema_eye_yaw:.3f} (Base: {baseline_eye_yaw:.3f}) Distracted: {is_eyes_away}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"Head Yaw: {ema_h_yaw:.2f} (Distracted: {is_head_away})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(image, f"EAR: {ema_ear:.2f} (Closed: {is_eyes_closed})", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 8. Draw Landmarks (similar to example)
            for lm_idx in [1, 234, 454, 473, 468, 33, 133, 362, 263]:
                lm = face_landmarks.landmark[lm_idx]
                cv2.circle(image, (int(lm.x * img_w), int(lm.y * img_h)), radius=2, color=(255, 0, 0), thickness=-1)

        # 3. Time Windows Update (30s window) - ALWAYS RUN
        head_away_history.append((curr_time, is_head_away))
        eyes_away_history.append((curr_time, is_eyes_away))
        
        while head_away_history and curr_time - head_away_history[0][0] > 30:
            head_away_history.popleft()
        while eyes_away_history and curr_time - eyes_away_history[0][0] > 30:
            eyes_away_history.popleft()
            
        cum_head_away = calc_cumulative_time(head_away_history)
        cum_eyes_away = calc_cumulative_time(eyes_away_history)

        # 4. Continuous Event Trackers
        if is_head_away:
            head_forward_start = None
        else:
            last_head_forward_time = curr_time
            if head_forward_start is None:
                head_forward_start = curr_time
                
        if is_eyes_away:
            eyes_forward_start = None
        else:
            last_eyes_forward_time = curr_time
            if eyes_forward_start is None:
                eyes_forward_start = curr_time
                
        if is_eyes_closed:
            eyes_open_start = None
        else:
            last_eyes_open_time = curr_time
            if eyes_open_start is None:
                eyes_open_start = curr_time

        # 5. Distraction State Machine Logic
        # Sleep
        if (curr_time - last_eyes_open_time) >= 7:
            state_sleep = True
            state_microsleep = False
        elif state_sleep and eyes_open_start and (curr_time - eyes_open_start) >= 2:
            state_sleep = False
            
        # Microsleep
        if not state_sleep:
            if (curr_time - last_eyes_open_time) >= 4:
                state_microsleep = True
            elif state_microsleep and eyes_open_start and (curr_time - eyes_open_start) >= 2:
                state_microsleep = False

        # Long Owl
        if (curr_time - last_head_forward_time) >= 5:
            state_long_owl = True
        elif state_long_owl and not is_head_away: 
            state_long_owl = False

        # Short Owl
        if state_short_owl:
            # Reset if looking forward for >= 2 seconds
            if head_forward_start and (curr_time - head_forward_start) >= 2:
                state_short_owl = False
                head_away_history.clear() # Clear window to prevent immediate re-trigger
        elif cum_head_away >= 10:
            state_short_owl = True

        # Long Lizard
        if (curr_time - last_eyes_forward_time) >= 5:
            state_long_lizard = True
        elif state_long_lizard and not is_eyes_away:
            state_long_lizard = False

        # Short Lizard
        if state_short_lizard:
            # Reset if looking forward for >= 2 seconds
            if eyes_forward_start and (curr_time - eyes_forward_start) >= 2:
                state_short_lizard = False
                eyes_away_history.clear() # Clear window to prevent immediate re-trigger
        elif cum_eyes_away >= 10:
            state_short_lizard = True

        # 6. Prioritize Displayed State
        if not is_face_detected:
            driver_state = "Face not detected"
        elif state_sleep:
            driver_state = "Sleep"
        elif state_microsleep:
            driver_state = "Microsleep"
        elif state_long_owl or state_long_lizard:
            driver_state = "Distracted (long)"
        elif state_short_owl or state_short_lizard:
            driver_state = "Distracted (short)"
        else:
            driver_state = "Focused on the road"
            
        # 7. Compute HR (Update every 1 sec)
        if is_face_detected and curr_time - last_hr_update > 1.0:
            if len(time_buffer) > 100:
                est_fps = len(time_buffer) / (time_buffer[-1] - time_buffer[0])
                current_hr = estimate_hr_rPPG(list(r_buffer), list(g_buffer), list(b_buffer), est_fps)
            last_hr_update = curr_time

        # --- Render Overlays ---
        # Bottom Right: Driver State
        text_size = cv2.getTextSize(driver_state, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = img_w - text_size[0] - 20
        text_y = img_h - 20
        
        # Color coding: Green if focused, Yellow for face missing, Red for distractions
        if driver_state == "Focused on the road":
            color = (0, 255, 0)
        elif driver_state == "Face not detected":
            color = (0, 255, 255) # Yellow
        else:
            color = (0, 0, 255) # Red
            
        cv2.putText(image, driver_state, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Bottom Left: Heart Rate
        hr_text = f"HR: {int(current_hr)} BPM" if current_hr > 0 else "HR: Calculating..."
        cv2.putText(image, hr_text, (20, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('DMS Output', image)

        if cv2.waitKey(5) & 0xFF == 27: # ESC
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
