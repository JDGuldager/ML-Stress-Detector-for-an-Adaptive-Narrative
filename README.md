# Adaptive Narrative Using Real-Time Stress Detection

This project explores how real-time physiological signals can be used to drive adaptive storytelling in a game.  
A machine learning model estimates the player’s stress state from wearable sensor data and feeds this information into an Unreal Engine experience, where it influences dialogue, animation, and narrative outcomes.

---

## What This Project Does

- Reads electrodermal activity (EDA) and blood volume pulse (BVP) from a wrist or hand worn sensor
- Uses a machine learning model to classify the user state as baseline, relaxed, or stressed
- Streams the predicted state into Unreal Engine in real time
- Combines physiological state with player dialogue choices to adapt the narrative

This is not a medical system. The goal is to explore physiological input as a narrative mechanic.

---

## How the Model Was Made

- Trained on the WESAD dataset using wrist based EDA and BVP signals
- Sensor data split into short time windows
- Simple time domain features extracted from EDA and heart related signals
- Subject specific baseline normalization applied to reduce physiological differences
- Random Forest classifier trained using subject independent cross validation
- Trained model exported and used for real time inference

---

## Real-Time System

- Live sensor data is streamed using Lab Streaming Layer (LSL)
- Python scripts handle preprocessing, baseline calibration, feature extraction, and prediction
- Unreal Engine Blueprints receive the predicted state and use it to drive game logic
- States and player choices are logged for later evaluation

---

## Tools Used

- Python (NumPy, SciPy, scikit-learn, joblib)
- WESAD dataset
- Lab Streaming Layer (LSL)
- Unreal Engine 5.6
- MetaHuman
- PsychoPy (baseline interface)
- OpenSignals / BioBridge (sensor streaming)

---

## Videos

▶️ **How the Machine Learning Model Works**  
https://youtu.be/_JsQaANyR3A

▶️ **Gameplay and Adaptive Narrative Demo**  
https://youtu.be/5H_EJY0WbgA

---

## Notes

- Model performance was evaluated offline on WESAD
- The integrated real-time system was tested qualitatively
- Wrist based physiological signals are noisy, so results should be interpreted accordingly

---

If you have questions or want to build on this, feel free to explore the code or reach out.
