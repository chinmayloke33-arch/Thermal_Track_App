import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    min_scale_temp: float = 7.0,
    max_scale_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Analyzes thermal images by extracting lightness intensity across HSV color space.
    """
    # 1. Convert RGB to HSV to capture hue and value (brightness) accurately
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Extract Value (brightness) and Saturation components
    v_channel = hsv[:, :, 2].astype(np.float32)
    s_channel = hsv[:, :, 1].astype(np.float32)

    # Combine Value and Saturation to avoid colorbar text or dark background artifacts
    thermal_intensity = v_channel * (s_channel / 255.0)

    # 2. Extract relative pixel intensity range (excluding top/bottom 1% extremes)
    p_min = np.percentile(thermal_intensity, 1)
    p_max = np.percentile(thermal_intensity, 99)

    if p_max <= p_min:
        raise ValueError("Image lacks sufficient thermal variation for analysis.")

    # 3. Normalize intensity mapping
    normalized = np.clip((thermal_intensity - p_min) / (p_max - p_min), 0.0, 1.0)

    # 4. Map to target temperature range
    temperature_map = min_scale_temp + normalized * (max_scale_temp - min_scale_temp)

    # 5. Calculate statistics from valid region (ignoring background black padding)
    mask = thermal_intensity > np.percentile(thermal_intensity, 5)
    valid_temps = temperature_map[mask] if np.any(mask) else temperature_map

    min_temp = float(np.min(valid_temps))
    max_temp = float(np.max(valid_temps))
    mean_temp = float(np.mean(valid_temps))
    temp_diff = max_temp - min_temp

    # 6. Fault assessment logic
    if temp_diff < threshold:
        status = "NO FAULT"
        action = "No immediate action required."
    else:
        status = "FAULT"
        action = "Attention required within 2 days."

    return {
        "min_temperature": min_temp,
        "max_temperature": max_temp,
        "temperature_difference": temp_diff,
        "mean_temperature": mean_temp,
        "threshold": threshold,
        "status": status,
        "action": action
    }
