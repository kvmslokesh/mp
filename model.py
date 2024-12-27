# model.py
import random

def detect_attack(data):
    # Simulate model prediction
    is_attack = random.choice([True, False])
    confidence = round(random.uniform(0.7, 1.0), 2) if is_attack else round(random.uniform(0.3, 0.6), 2)
    print(confidence)
    return {"is_attack": is_attack, "confidence": confidence}
