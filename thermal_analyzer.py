import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    min_scale_temp: float = 7.0,
    max_scale_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Analyzes relative thermal variations by focusing on active regions 
    and avoiding fixed min/max range locking.
    """
    # 1. Convert to grayscale/intensity map
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 2. Focus analysis on central 80% to ignore border legends, text, and scales
    h, w = gray.shape
    roi = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

    # 3. Use standard deviation & distribution percentiles instead of absolute min/max
    mean_val = np.mean(roi)
    std_val = np.std(roi)

    # Define baseline temperature (ambient) and peak temperature (hotspot)
    low_val = np.clip(mean_val - (1.5 * std_val), np.min(roi), np.max(roi))
    high_val = np.clip(mean_val + (2.5 * std_val), np.min(roi), np.max(roi))

    # 4. Map pixel values to temperature scale range
    scale_span = max_scale_temp - min_scale_temp
    
    min_temp = float(min_scale_temp + (low_val / 255.0) * scale_span)
    max_temp = float(min_scale_temp + (high_val / 255.0) * scale_span)
    mean_temp = float(min_scale_temp + (mean_val / 255.0) * scale_span)
    
    # Dynamic temperature difference based on subject hotspot vs ambient
    temp_diff = float(max_temp - min_temp)

    # 5. Fault decision logic
    if temp_diff < threshold:
        status = "NO FAULT"
        action = "No immediate action required."
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
