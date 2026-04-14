import cv2
import numpy as np
import sys
import glob
import os

class Config:
    """
    Configuration parameters based on the GOLD paper and Assignment 2.
    """
    # Camera parameters
    IMAGE_W = 1920
    IMAGE_H = 1080
    U0 = 970
    V0 = 483
    FX = 1970
    FY = 1970
    H = 1.66       # Camera height in meters
    PITCH = 0.0    # Pitch angle in radians
    
    Z_MIN = 6.0    
    Z_MAX = 25.0   # Range: 19 metri (da z=6 a z=25)
    X_MIN = -2.5   
    X_MAX = 2.5    # Range: 5 metri (da x=-2.5 a x=2.5)
    
    # BEV Isotropic
    BEV_W = 125   
    BEV_H = 475   
    
    # Filter parameters
    M = 3          # Horizontal pixel distance for lane width filter
    C = 15         # Neighborhood size for adaptive thresholding
    K = 1.5        # Constant for adaptive thresholding
    
    # Confidence threshold for lane extraction
    LANE_CONFIDENCE = 40 * 255 # Minimum sum of pixels to consider a lane
    
    # Noise floor: global minimal threshold to suppress starburst artifacts on dark asphalt
    TAU = 12 


# IPM: Inverse Perspective Mapping
def get_perspective_transformation(cfg):
    """Compute the IPM homography from camera parameters and ROI definition."""
    def project_to_image(X, Z):
        u = (cfg.FX * X / Z) + cfg.U0
        v = (cfg.FY * cfg.H / Z) + cfg.V0
        return [u, v]

    # Project the four ROI corners from road plane to image plane
    bl = project_to_image(cfg.X_MIN, cfg.Z_MIN)
    br = project_to_image(cfg.X_MAX, cfg.Z_MIN)
    tr = project_to_image(cfg.X_MAX, cfg.Z_MAX)
    tl = project_to_image(cfg.X_MIN, cfg.Z_MAX)

    src_pts = np.float32([bl, br, tr, tl])

    dst_pts = np.float32([
        [0, cfg.BEV_H],
        [cfg.BEV_W, cfg.BEV_H],
        [cfg.BEV_W, 0],
        [0, 0]
    ])

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    inv_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
    roi_polygon = np.array(src_pts, np.int32).reshape((-1, 1, 2))

    return matrix, inv_matrix, roi_polygon

# Helper to map a BEV pixel back to original perspective image
def bev_to_persp(x_bev, y_bev, inv_matrix):
    """Helper to map a BEV pixel back to original perspective image"""
    pt = np.array([[[x_bev, y_bev]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, inv_matrix)[0][0]
    return int(res[0]), int(res[1])

# Horizontal brightness transition filter
def apply_horizontal_filter(bev_gray, m):
    """
    Applies the horizontal brightness transition filter.
    It extracts vertical bright structures bounded by darker regions.
    """
    img = bev_gray.astype(np.float32)
    
    # Shift images to get left and right neighbors at distance m
    img_left = np.zeros_like(img)
    img_right = np.zeros_like(img)
    
    img_left[:, m:] = img[:, :-m]   # b(x, y-m)
    img_right[:, :-m] = img[:, m:]  # b(x, y+m)
    
    # Differences
    d_plus = img - img_right
    d_minus = img - img_left
    
    # Condition: bright center, dark left and right
    mask = (d_plus > 0) & (d_minus > 0)
    
    filtered = np.zeros_like(img)
    filtered[mask] = d_plus[mask] + d_minus[mask]
    
    return filtered

# Iterative thresholding
def prof_threshold(filtered_img, Tau):
    """
    Iterative thresholding based on the formulas,
    optimized to ignore background micro-noise and handle peaks.
    """
    # 1. PRE-CLEANING: Ignore very low intensity noise (e.g., values below 10).
    # This prevents thousands of almost black pixels from destroying the g_B mean.
    active_pixels = filtered_img[filtered_img > 10]
    
    # If the image is completely dark (e.g., empty asphalt), return black
    if len(active_pixels) == 0:
        return np.zeros_like(filtered_img, dtype=np.uint8)
        
    # 2. SMART INITIALIZATION: Lane lines are very bright but few.
    # Start from the 75th percentile instead of the mean for faster and more stable convergence.
    Th = np.percentile(active_pixels, 75)
    
    # Iterative cycle to find the optimal threshold
    for _ in range(30): # Slightly increased the iteration margin
        # Separate pixels into two groups based on the current threshold
        group_A = active_pixels[active_pixels >= Th]
        group_B = active_pixels[active_pixels < Th]
        
        # 3. SAFETY: Calculate g_A and g_B. If a group is empty, use Th to avoid throwing off the formula.
        g_A = np.mean(group_A) if len(group_A) > 0 else Th
        g_B = np.mean(group_B) if len(group_B) > 0 else Th
        
        # New threshold: the mean between the two groups
        new_Th = (g_A + g_B) / 2.0
        
        # Exit condition: if the threshold changes by less than 1 gray level, we are done.
        if abs(new_Th - Th) < 1.0:
            Th = new_Th
            break
            
        Th = new_Th
        
    # 4. NOISE FLOOR: global minimal threshold to suppress starburst artifacts on dark asphalt
    if Th < Tau:
        return np.zeros_like(filtered_img, dtype=np.uint8)
    # Apply the final calculated threshold to create the binary image
    binary = np.zeros_like(filtered_img, dtype=np.uint8)
    binary[filtered_img >= Th] = 255
    
    return binary

def extract_lanes(binary_img, cfg):
    # 1. REMOVE NOISE AT THE EDGES: 
    # The black borders of the IPM create false positives. We zero them out brutally (15% of the screen on the right and left).
    crop_margin = int(binary_img.shape[1] * 0.15)
    binary_img[:, :crop_margin] = 0
    binary_img[:, -crop_margin:] = 0

    # Generate a column-wise sum (histogram) of the bottom half of the image
    histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
    
    # 2. DYNAMIC MARGIN: We widen the central "blind zone" to skip the writings on the asphalt
    midpoint = binary_img.shape[1] // 2
    search_offset = 18  # Ignore the central zone
    
    left_lane_x = None
    right_lane_x = None
    
    # --- Search for the left lane (from the clean zone to the center) ---
    left_hist = histogram[:midpoint - search_offset]
    if len(left_hist) > 0:
        max_val_left = np.max(left_hist)
        # Lower the confidence slightly to be sure to get it
        if max_val_left > (20 * 255): 
            left_lane_x = np.argmax(left_hist)
            
    # --- Search for the right lane (from the center to the clean zone) ---
    right_hist = histogram[midpoint + search_offset:]
    if len(right_hist) > 0:
        max_val_right = np.max(right_hist)
        if max_val_right > (20 * 255): 
            right_lane_x = np.argmax(right_hist) + midpoint + search_offset
            
    return left_lane_x, right_lane_x


# Draw histogram
def draw_histogram(binary_img, cfg):
    """
    Creates a visual representation of the lane detection histogram.
    """
    histogram = np.sum(binary_img[binary_img.shape[0]//2:, :], axis=0)
    hist_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), dtype=np.uint8)
    
    # Determine scale based on max value or threshold
    max_val = max(np.max(histogram), cfg.LANE_CONFIDENCE * 1.5)
    if max_val == 0: max_val = 1
    
    # Draw the histogram bars
    for x, val in enumerate(histogram):
        h = int((val / max_val) * (hist_img.shape[0] - 20))
        if h > 0:
            cv2.line(hist_img, (x, hist_img.shape[0]), (x, hist_img.shape[0] - h), (255, 255, 255), 1)
    
    cv2.putText(hist_img, "Histogram Peak Search", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    return hist_img

# Optional 1: Lane Type Classification
def classify_lanes(binary_img, left_x, right_x):
    """
    Optional 1: Lane Type Classification.
    Classify each detected lane as CONTINUOUS or DASHED based on vertical continuity.
    """
    def classify_single(x):
        if x is None:
            return None
            
        # Extract a narrow strip around the lane x-coordinate
        start_x = max(0, x - 3)
        end_x = min(binary_img.shape[1], x + 4)
        
        # Analyze the lower half of the BEV (about 9.5 meters of real road)
        lane_strip = binary_img[binary_img.shape[0]//2:, start_x:end_x]
        
        # Flatten horizontally
        profile = np.max(lane_strip, axis=1) > 0
        
        # Calculate heuristics
        fill_ratio = np.sum(profile) / len(profile)
        transitions = np.sum(profile[:-1] != profile[1:])
        
        # Logic: 
        # - In 9.5m, we might see only 1 interruption (2 or 3 transitions).
        # - The fill_ratio of a dashed line will be around 0.3 - 0.5. 
        #   If it's < 0.85 we consider that there are evident holes.
        if transitions >= 2 and fill_ratio < 0.65:
            return "DASHED"
        else:
            return "CONTINUOUS"

    return classify_single(left_x), classify_single(right_x)

    
def main():
    # 1. Parse CLI arguments
    if len(sys.argv) != 2:
        print("Usage: python3 run_gold.py '<path_to_images>'")
        print("Example: python3 run_gold.py 'PandaSetSensorData/archive/044/Camera/front_camera/*.jpg'")
        sys.exit(1)
        
    search_path = sys.argv[1]
    image_paths = sorted(glob.glob(search_path))
    
    if not image_paths:
        print(f"No images found for path: {search_path}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images. Press any key to advance to the next frame. Press 'q' or 'ESC' to quit.")
    
    # Initialization Multi-Layer: Homography and Neural Network YOLO
    cfg = Config()
    perspective_matrix, inv_matrix, roi_polygon = get_perspective_transformation(cfg)
    
    # Initialize YOLO (Fast and super light, the framework will auto-download if not present)
    print("Initializing YOLOV8 in progress for Artificial Intelligence (Obstacles)...")
    from ultralytics import YOLO
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR) # silence output
    yolo_model = YOLO("yolov8n.pt") 
    
    # Variables for the temporal memory of the lanes (up to 3 frames of darkness)
    left_mem_x = None
    left_mem_type = None
    left_miss_cnt = 0
    
    right_mem_x = None
    right_mem_type = None
    right_miss_cnt = 0
    
    # 3. Execution Loop
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        # -- Step A: Extract Region of Interest (ROI) and Remap to BEV --
        
        # Warp to bird's-eye view tramite Homography Matrix
        bev_img = cv2.warpPerspective(frame, perspective_matrix, (cfg.BEV_W, cfg.BEV_H))
        bev_gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
        
        # Draw the trapezoid on the original real image using the footprint calculated a priori
        cv2.polylines(frame, [roi_polygon], isClosed=True, color=(0, 0, 255), thickness=4)
        pt_TL = tuple(roi_polygon[3][0])
        cv2.putText(frame, "ROI", (pt_TL[0] - 50, pt_TL[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # -- Step B: Low-Level Processing (Parallel Feature Extraction) --
        filtered = apply_horizontal_filter(bev_gray, cfg.M)
        binary = prof_threshold(filtered, cfg.TAU)
        
        # -- Step C: Medium-Level Processing with Memory --
        left_x, right_x = extract_lanes(binary, cfg)
        left_t, right_t = classify_lanes(binary, left_x, right_x)
        
        # Memory update for Left lane
        if left_x is not None:
            left_mem_x = left_x
            left_mem_type = left_t
            left_miss_cnt = 0
        else:
            left_miss_cnt += 1
            if left_miss_cnt <= 4 and left_mem_x is not None:
                left_x = left_mem_x
                left_t = left_mem_type
            else:
                left_mem_x = None
                
        # Memory update for Right lane
        if right_x is not None:
            right_mem_x = right_x
            right_mem_type = right_t
            right_miss_cnt = 0
        else:
            right_miss_cnt += 1
            if right_miss_cnt <= 4 and right_mem_x is not None:
                right_x = right_mem_x
                right_t = right_mem_type
            else:
                right_mem_x = None
                
        # Recompose lists for downstream visualizations
        lanes_x = []
        lane_types = []
        if left_x is not None:
            lanes_x.append(left_x)
            lane_types.append(left_t)
        if right_x is not None:
            lanes_x.append(right_x)
            lane_types.append(right_t)
        
        # -- Initialization for Visualization --
        bev_out = bev_img.copy()
        
        # -- Optional 3: Obstacle Detection & Distance Estimation with YOLO (AI) --
        # Perform inference using YOLO only on Relevant classes in the street (people, bikes, cars, motorcycles, buses, trucks)
        results = yolo_model(frame, verbose=False, classes=[0, 1, 2, 3, 5, 7])
        
        for box in results[0].boxes:
            # Extract geometries and features from the Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label_name = yolo_model.names[cls_id]
            
            # Find the contact point with the ground (Lower central base of the bounding box)
            u_center = (x1 + x2) // 2
            v_bottom = y2
            
            # If it is above the perspective horizon it is irrelevant and would give asymptote
            if v_bottom <= cfg.V0 + 5:
                continue
                
            # Use homography to map (u_center, v_bottom) to the Bird's Eye view
            pt_to_map = np.array([[[u_center, v_bottom]]], dtype=np.float32)
            bev_pt = cv2.perspectiveTransform(pt_to_map, perspective_matrix)[0][0]
            bev_x, bev_y = int(bev_pt[0]), int(bev_pt[1])
            
            # If the obstacle falls within our complete ROI / BEV
            if 0 <= bev_x <= cfg.BEV_W and 0 <= bev_y <= cfg.BEV_H:
                
                # Calculate the distance Z (meters) in a native linear way,
                # exploiting the vertical axis of our Bird's Eye View scaled to the ground.
                distance = cfg.Z_MAX - (bev_y / cfg.BEV_H) * (cfg.Z_MAX - cfg.Z_MIN)
                
                # --- Drawing on Original Perspective ---
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label_str = f"[{label_name}] {distance:.1f}m"
                cv2.putText(frame, label_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # --- Drawing on BEV Plan (Highlighting the radial contact on the ground) ---
                cv2.circle(bev_out, (bev_x, bev_y), 8, (0, 0, 255), -1)
                cv2.putText(bev_out, f"{distance:.1f}m", (bev_x + 12, bev_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # -- Optional 2: Perspective Overlay & True Dashed Generation --
        for x, ltype in zip(lanes_x, lane_types):
            u_text, v_text = bev_to_persp(x, cfg.BEV_H - 50, inv_matrix)
            cv2.putText(frame, ltype, (u_text + 30, v_text), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
            
            if ltype == "DASHED":
                # Scan the row to mark 'DASHED' only where there is the actual road paint
                for y in range(0, cfg.BEV_H):
                    if np.max(binary[y, max(0, x-2):min(cfg.BEV_W, x+3)]) > 0:
                        u, v = bev_to_persp(x, y, inv_matrix)
                        cv2.circle(frame, (u, v), 3, (0, 255, 0), -1)
                        cv2.circle(bev_out, (x, y), 2, (0, 255, 0), -1)
            else:
                u_top, v_top = bev_to_persp(x, 0, inv_matrix)
                u_bottom, v_bottom = bev_to_persp(x, cfg.BEV_H - 1, inv_matrix)
                
                cv2.line(frame, (u_bottom, v_bottom), (u_top, v_top), (0, 255, 0), 6)
                cv2.line(bev_out, (x, 0), (x, bev_out.shape[0]), (0, 255, 0), 3)
        
        # -- Step D: Mandatory Visualization (Dashboard Integration) --
        
        if len(lanes_x) == 0:
            text = "No lanes found"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_y = frame.shape[0] - 50 
            
            text_x_frame = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, text, (text_x_frame, text_y), font, font_scale, (0, 0, 255), thickness)
                
        # Dashboard creation: Horizontal Layout (Side-by-side)
        dashboard_h = 600 # Fixed height to 600 pixels
        
        # 1. Original frame (Maintains 16:9 proportions)
        orig_w = int(frame.shape[1] * (dashboard_h / frame.shape[0]))
        frame_resized = cv2.resize(frame, (orig_w, dashboard_h))
        cv2.putText(frame_resized, "Original Perspective", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 2. BEV out (Narrow and long)
        bev_w = int(bev_out.shape[1] * (dashboard_h / bev_out.shape[0]))
        bev_resized = cv2.resize(bev_out, (bev_w, dashboard_h))
        cv2.putText(bev_resized, "BEV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3. Binary BGR (Narrow and long)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        binary_resized = cv2.resize(binary_bgr, (bev_w, dashboard_h))
        cv2.putText(binary_resized, "Binary", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 4. Histogram (Narrow and long)
        hist_img = draw_histogram(binary, cfg)
        hist_resized = cv2.resize(hist_img, (bev_w, dashboard_h))
        
        # Join everything horizontally in a single long row
        dashboard = np.hstack((frame_resized, bev_resized, binary_resized, hist_resized))
        
        # Avoid the window being truncated by the operating system (e.g. 150% scaling on laptop)
        max_screen_width = 1500
        if dashboard.shape[1] > max_screen_width:
            scale = max_screen_width / dashboard.shape[1]
            dashboard = cv2.resize(dashboard, (int(dashboard.shape[1] * scale), int(dashboard.shape[0] * scale)))
        
        # Show dashboard
        cv2.namedWindow("Dashboard - GOLD Lane Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Dashboard - GOLD Lane Detection", dashboard)
        
        # Replace zero (which blocks the frame indefinitely until clicked)
        # with '30' which scrolls smoothly at ~33 frames per second.
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            break


if __name__ == '__main__':
    main()