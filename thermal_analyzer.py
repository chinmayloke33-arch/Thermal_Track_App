import cv2
import numpy as np


def analyze_thermal_image(
    full_image: np.ndarray,
    cropped_roi: np.ndarray,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Uses global image intensity bounds to set the temperature scale slope, 
    then calculates actual local temperatures within the user's cropped ROI.
    """
    # 1. Convert full image to grayscale to find global brightness bounds
    full_gray = cv2.cvtColor(full_image, cv2.COLOR_RGB2GRAY)
    
    # Strip extreme 2% outer border to ignore camera text / UI overlay
    h, w = full_gray.shape
    inner_full = full_gray[int(h * 0.02):int(h * 0.98), int(w * 0.02):int(w * 0.98)]
    
    global_min_pixel = float(np.min(inner_full))
    global_max_pixel = float(np.max(inner_full))

    if global_max_pixel <= global_min_pixel:
        global_max_pixel = 255.0
        global_min_pixel = 0.0

    # True scaling factor (°C per pixel brightness step) across the camera's frame
    temp_per_pixel = (known_hot_temp - known_cold_temp) / (global_max_pixel - global_min_pixel)

    # 2. Convert cropped ROI to grayscale and read local intensities
    roi_gray = cv2.cvtColor(cropped_roi, cv2.COLOR_RGB2GRAY)
    
    roi_min_pixel = float(np.min(roi_gray))
    roi_max_pixel = float(np.max(roi_gray))
    roi_mean_pixel = float(np.mean(roi_gray))

    # 3. Map local ROI pixels using the true global scale
    min_temp = known_cold_temp + (roi_min_pixel - global_min_pixel) * temp_per_pixel
    max_temp = known_cold_temp + (roi_max_pixel - global_min_pixel) * temp_per_pixel
    mean_temp = known_cold_temp + (roi_mean_pixel - global_min_pixel) * temp_per_pixel

    # Clamp bounds to input min/max
    min_temp = max(known_cold_temp, min(known_hot_temp, min_temp))
    max_temp = max(known_cold_temp, min(known_hot_temp, max_temp))
    
    temp_diff = float(max_temp - min_temp)

    # 4. Fault classification
    if temp_diff < threshold:
        status = "NO FAULT"
        action = "Operating within normal thermal limits."
    else:
        status = "FAULT"
        action = "Attention required within 2 days."

    return {
        "min_temperature": round(min_temp, 2),
        "max_temperature": round(max_temp, 2),
        "temperature_difference": round(temp_diff, 2),
        "mean_temperature": round(mean_temp, 2),
        "threshold": threshold,
        "status": status,
        "action": action
    }
