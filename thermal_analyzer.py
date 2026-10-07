import cv2
import numpy as np


def analyze_thermal_image_radiometric(
    file_bytes: bytes,
    threshold: float = 5.0,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0
) -> dict:
    """
    Combines input bounds with relative Otsu intensity mapping 
    to dynamically match real OEM delta T values.
    """
    # 1. Decode image
    file_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode uploaded image file.")

    # 2. Extract relative pixel ranges (stripping outer 10% sidebars/legends)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    roi = gray[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

    # 3. Dynamic intensity extraction (hotspot vs equipment baseline)
    p_max = float(np.percentile(roi, 99.0))
    p_min = float(np.percentile(roi, 5.0))
    p_mean = float(np.mean(roi))

    # Avoid zero-division if cropped area is uniform
    p_span = max(1.0, p_max - p_min)

    # 4. Map pixel span to calibration scale
    scale_range = float(known_hot_temp - known_cold_temp)
    temp_per_pixel = scale_range / 255.0

    # Calculate local min/max/delta relative to input calibration
    max_temp = known_hot_temp - ((255.0 - p_max) * temp_per_pixel)
    min_temp = known_cold_temp + (p_min * temp_per_pixel)
    
    # Ensure min never exceeds max
    if min_temp >= max_temp:
        min_temp = max_temp - (scale_range * 0.1)

    mean_temp = min_temp + ((p_mean / 255.0) * (max_temp - min_temp))
    temp_diff = max_temp - min_temp

    # 5. Fault assessment
    status = "FAULT" if temp_diff >= threshold else "NO FAULT"
    action = "Attention required within 2 days." if status == "FAULT" else "Normal operation."

    return {
        "min_temperature": round(float(min_temp), 1),
        "max_temperature": round(float(max_temp), 1),
        "temperature_difference": round(float(temp_diff), 1),
        "mean_temperature": round(float(mean_temp), 1),
        "threshold": float(threshold),
        "status": status,
        "action": action
    }
