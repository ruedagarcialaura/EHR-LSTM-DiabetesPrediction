import pickle
import pandas as pd
import matplotlib.pyplot as plt

input_path = r"preprocessing\output_pickles\patients_filtered_unnormalized.pkl"


BINS_THRESHOLDS = {
    'ALT(SGPT)': [10, 49, 98, 245, 735],
    'AST(SGOT)': [10, 34, 68, 170, 510],
    'BUN': [6, 20, 50, 100, 250],
    'CHOLESTEROL': [200, 240],
    'HDL': [45, 80, 100],
    'HGB A1C': [5.7, 6.5],
    'POTASSIUM': [3.5, 5.1, 6.5],
    'TRIGLYCERIDE': [150, 200, 500],
    'VITAL_10541041_MEAN': [18.5, 25, 30, 35, 40],
    'VITAL_10541455_MEAN': [36.1, 37.3, 38.1, 39.1, 41.1],
    'VITAL_10541467_MEAN': [60, 100],
    'VITAL_10541503_MEAN': [12, 20],
    'VITAL_266705352_MEAN': [90, 92, 94, 97, 100],
    'VITAL_283305634_MEAN': [97.0, 99.1, 100.6, 102.4, 105.8],
    'VITAL_68924855_MEAN': [120, 130, 140, 180]
}


def analyze_and_plot_distribution(patient_data):
    """Calculates categorization coverage and plots the distribution."""
    # 1. Get the total number of unique features in the dataset
    first_pid = next(iter(patient_data))
    all_features_in_data = [idx[0] if isinstance(idx, tuple) else str(idx) 
                            for idx in patient_data[first_pid].index]
    
    total_feature_count = len(all_features_in_data)
    
    # 2. Identify which features in the data match our categorization dictionary
    dict_keys = set(BINS_THRESHOLDS.keys())
    matched_features = [f for f in all_features_in_data if f in dict_keys]
    
    categorized_count = len(matched_features)
    continuous_count = total_feature_count - categorized_count
    perc_categorized = (categorized_count / total_feature_count) * 100

    print("   DATA DISTRIBUTION STATS")
    print(f"Total Features in Data:  {total_feature_count}")
    print(f"Categorized Variables:   {categorized_count}")
    print(f"Continuous Variables:    {continuous_count}")
    print(f"Coverage Percentage:     {perc_categorized:.2f}%")

    # 3. Plotting    
    labels = ['Categorized (Clinical Bins)', 'Continuous (Z-Score)']
    sizes = [categorized_count, continuous_count]
    colors = ['#4CAF50', '#2196F3']

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{v:d}\n({p:.1f}%)'.format(v=val, p=pct)
        return my_autopct
    
    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct=make_autopct(sizes), startangle=140, 
            colors=colors, explode=(0.1, 0), shadow=True)
    plt.title(f'Feature Categorization Strategy\n(Total Rows: {total_feature_count})')
    plt.show()





def validate_names():
    try:
        with open(input_path, "rb") as f:
            patient_data = pickle.load(f)
        print(f" Data loaded successfully.")
    except Exception as e:
        print(f" Loading error: {e}")
        return

    # Extract all real features from the dataset
    all_real_features = set()
    for pid, df in patient_data.items():
        # Get the canonical name (first element if it's a tuple/MultiIndex)
        for idx in df.index:
            name = idx[0] if isinstance(idx, tuple) else str(idx)
            all_real_features.add(name)
    
    # Set-based comparison
    dict_keys = set(BINS_THRESHOLDS.keys())
    matched = dict_keys.intersection(all_real_features)
    missing = dict_keys - all_real_features
    
    print("FEATURE NAME VALIDATION RESULTS")
    print(f"Thresholds in Dictionary: {len(dict_keys)}")
    print(f"Matching Features Found:  {len(matched)}")
    print(f"Missing/Mismatched:       {len(missing)}")
    
    if missing:
        print("\n THE FOLLOWING KEYS DO NOT MATCH THE DATA:")
        for m in sorted(missing):
            print(f"   - '{m}'")
        print("First 20 actual names found in your dataset for reference:")
        for real in sorted(list(all_real_features))[:20]:
            print(f"   > '{real}'")
    else:
        print("\n Success! All dictionary keys match your dataset exactly.")
    
    analyze_and_plot_distribution(patient_data)

if __name__ == "__main__":
    validate_names()