def detect_drift(confidence, low=0.4, high=0.6):
    """
    Drift if model confidence is uncertain
    """
    if low <= confidence <= high:
        return True, "⚠️ Drift Detected (Uncertain Prediction)"
    else:
        return False, "✅ No Drift (Prediction Stable)"