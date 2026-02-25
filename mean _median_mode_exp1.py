# Get input from the user
raw_input = input("Enter numbers separated by spaces: ")

try:
    # Convert the string input into a list of floats
    data = [float(x) for x in raw_input.split()]

    if not data:
        print("The list is empty. Please enter some numbers.")
    else:
        # --- MEAN ---
        mean = sum(data) / len(data)

        # --- MEDIAN ---
        data.sort()  # Sorts the list in place
        n = len(data)
        if n % 2 == 1:
            median = data[n // 2]
        else:
            median = (data[n // 2 - 1] + data[n // 2]) / 2

        # --- MODE ---
        frequency = {}
        for item in data:
            frequency[item] = frequency.get(item, 0) + 1

        max_freq = max(frequency.values())
        # Handles multiple modes (multimodal)
        modes = [key for key, val in frequency.items() if val == max_freq]

        # --- DISPLAY RESULTS ---
        print("-" * 30)
        print(f"Sorted Data: {data}")
        print(f"Mean:   {mean:.2f}")
        print(f"Median: {median}")

        if max_freq == 1 and len(data) > 1:
            print("Mode:   No unique mode (all values appear once)")
        else:
            print(f"Mode:   {modes}")

except ValueError:
    print("Invalid input! Please enter only numbers separated by spaces.")