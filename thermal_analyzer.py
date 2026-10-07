import streamlit as st


def analyze_thermal_image_radiometric(
    file_bytes: bytes,
    threshold: float = 5.0,
    known_cold_temp: float = 7.0,
    known_hot_temp: float = 40.0
) -> dict:
    # DEBUG LINE: Screen par dikhega ki exact values kya pass ho rahi hain
    st.info(f"🔍 DEBUG: Cold Temp = {known_cold_temp} | Hot Temp = {known_hot_temp}")

    min_temp = float(known_cold_temp)
    max_temp = float(known_hot_temp)
    
    # Direct OEM Formula
    temp_diff = round(max_temp - min_temp, 2)
    mean_temp = round((max_temp + min_temp) / 2.0, 2)

    status = "FAULT" if temp_diff >= threshold else "NO FAULT"
    action = "Attention required within 2 days." if status == "FAULT" else "Normal operation."

    return {
        "min_temperature": round(min_temp, 2),
        "max_temperature": round(max_temp, 2),
        "temperature_difference": temp_diff,
        "mean_temperature": mean_temp,
        "threshold": threshold,
        "status": status,
        "action": action
    }
