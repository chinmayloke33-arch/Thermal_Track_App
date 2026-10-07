import cv2
import numpy as np


def analyze_thermal_image(
    image: np.ndarray,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0,
    threshold: float = 5.0
) -> dict:
    """
    Dynamically extracts hotspot vs healthy component baseline across diverse
    thermal image palettes to maintain a precise, consistent ΔT reading.
    """
    # 1. Convert to grayscale intensity map
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 2. Focus on central target region (ignoring outer graphics and legends)
    h, w = gray.shape
    roi = gray[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)]

    # 3. Apply bilateral filter (preserves sharp hotspot edges while smoothing noise)
    filtered = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Filter out cold ambient background (keep top 40% brightest regions)
    background_cutoff = np.percentile(filtered, 60)
    component_pixels = filtered[filtered >= background_cutoff]

    if len(component_pixels) == 0:
        component_pixels = filtered.flatten()

    # 5. Dynamic Baseline Calculation:
    # Baseline = Mode (most frequent temperature intensity) of the active equipment
    counts, bin_edges = np.histogram(component_pixels, bins=30)
    max_bin_index = np.argmax(counts)
    baseline_pixel = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2.0

    # 6. Peak Hotspot Calculation:
    # Hotspot = Average of the top 0.5% hottest pixels
    hotspot_cutoff = np.percentile(component_pixels, 99.5)
    hotspot_pixels = component_pixels[component_pixels >= hotspot_cutoff]
    hotspot_pixel = np.mean(hotspot_pixels) if len(hotspot_pixels) > 0 else np.max(component_pixels)

    # 7. Convert intensities to actual Celsius scale
    scale_range = known_hot_temp - known_cold_temp
    temp_per_pixel = scale_range / 255.0

    max_temp = float(known_cold_temp + (hotspot_pixel * temp_per_pixel))
    min_temp = float(known_cold_temp + (baseline_pixel * temp_per_pixel))
    mean_temp = float(known_cold_temp + (np.mean(component_pixels) * temp_per_pixel))

    # Calculate precise hotspot vs healthy component Delta T
    temp_diff = float(max_temp - min_temp)

    # 8. Fault assessment logic
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
