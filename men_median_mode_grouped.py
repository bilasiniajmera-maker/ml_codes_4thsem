import pandas as pd

try:
    num_classes = int(input("Enter the number of class intervals: "))
    classes_data = []

    print("\nEnter details for each class (Lower Limit, Upper Limit, Frequency):")
    for i in range(num_classes):
        line = input(f"Class {i+1} (e.g., 10 20 5): ").split()
        l, u, f = map(float, line)
        classes_data.append({'Lower Limit': l, 'Upper Limit': u, 'Frequency': f})

    df_grouped = pd.DataFrame(classes_data)

    # Calculate Midpoint and Cumulative Frequency
    df_grouped['Midpoint'] = (df_grouped['Lower Limit'] + df_grouped['Upper Limit']) / 2
    df_grouped['Cumulative Frequency'] = df_grouped['Frequency'].cumsum()

    # Total frequency (n) and class width (h)
    n = df_grouped['Frequency'].sum()
    h = df_grouped['Upper Limit'].iloc[0] - df_grouped['Lower Limit'].iloc[0] # Assumes uniform class width

    # --- 1. MEAN (using pandas for weighted average) ---
    mean = (df_grouped['Midpoint'] * df_grouped['Frequency']).sum() / n

    # --- 2. MEDIAN (formula implementation) ---
    median_class_index = df_grouped[df_grouped['Cumulative Frequency'] >= n / 2].index[0]
    median_class = df_grouped.loc[median_class_index]

    l_med = median_class['Lower Limit']
    f_med = median_class['Frequency']
    prev_cf = df_grouped['Cumulative Frequency'].iloc[median_class_index - 1] if median_class_index > 0 else 0

    median = l_med + (((n / 2) - prev_cf) / f_med) * h

    # --- 3. MODE (formula implementation) ---
    modal_class_index = df_grouped['Frequency'].idxmax()
    modal_class = df_grouped.loc[modal_class_index]

    l_mod = modal_class['Lower Limit']
    f1 = modal_class['Frequency']
    f0 = df_grouped['Frequency'].iloc[modal_class_index - 1] if modal_class_index > 0 else 0
    f2 = df_grouped['Frequency'].iloc[modal_class_index + 1] if modal_class_index < len(df_grouped) - 1 else 0

    denominator = (2 * f1 - f0 - f2)
    mode = l_mod + ((f1 - f0) / denominator) * h if denominator != 0 else l_mod # Handle division by zero

    # --- OUTPUT ---
    print("\n" + "="*30)
    print(f"{'STATISTIC':<15} | {'VALUE':<10}")
    print("-" * 30)
    print(f"{'Mean':<15} | {mean:.2f}")
    print(f"{'Median':<15} | {median:.2f}")
    print(f"{'Mode':<15} | {mode:.2f}")
    print("="*30)

except Exception as e:
    print(f"Error: Please ensure you enter numbers correctly. ({e})")