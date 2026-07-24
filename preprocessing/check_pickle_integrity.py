"""
Quick check: is preprocessing/output_pickles/2b_lab_vitals_dx.pkl actually a
valid, complete pickle file, or did it get truncated by the MemoryError?
"""
import pickle

path = r"preprocessing\output_pickles\2b_lab_vitals_dx.pkl"

try:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        print(f"VALID (old-style): loaded a dict with {len(data)} patients.")
    else:
        print(f"VALID (new-style): first object is a count = {data}")
except Exception as e:
    print(f"CORRUPTED / INCOMPLETE: {type(e).__name__}: {e}")
    print("This file cannot be trusted -- you'll need to re-run 2b_Data_Preprocessing_DX.py "
          "with the fixed (incremental-save) version once I finish updating it.")
