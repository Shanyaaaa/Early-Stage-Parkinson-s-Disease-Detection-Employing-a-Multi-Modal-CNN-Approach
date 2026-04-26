# import numpy as np
# import joblib

# # Load model
# clf = joblib.load("parkinson_classifier.pkl")
# reg = joblib.load("parkinson_stage_model.pkl")
# features = joblib.load("features.pkl")

# # Example patient input (CHANGE THIS)
# # patient = {
# #     "Age": 30,
# #     "UPDRS III": 0,
# #     "Disease Duration": 0,
# #     "Speech Score": 0,
# #     "Facial Expression": 0,
# #     "Tremor (Head)": 0,
# #     "Levodopa (mg/day)": 0
# # }

# # Convert to model input
# input_data = [patient[f] for f in features]
# input_array = np.array(input_data).reshape(1, -1)

# # Predict
# status = clf.predict(input_array)[0]
# stage = reg.predict(input_array)[0]

# print("\n--- RESULT ---")
# print("Status:", "Parkinson's" if status == 1 else "Healthy")
# print("Stage:", round(stage, 2))