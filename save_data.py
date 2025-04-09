import csv

def save_data_to_csv(file_path, data_dict):
    """
    Saves data from a dictionary to a CSV file.
    Each key in the dictionary becomes a column header, and the corresponding values are rows.
    """
    # Extract headers and rows
    headers = list(data_dict.keys())
    rows = zip(*data_dict.values())

    # Write to CSV
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(headers)
        # Write rows
        writer.writerows(rows)

def read_csv_to_dict(filename):
    """
    Reads a CSV file and returns a dictionary with keys and values from the first two columns.
    Assumes the first column contains the keys and the second column contains the values.
    """
    result = {}
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        # Automatically get the column names
        headers = reader.fieldnames
        key_col, value_col = headers[0], headers[1]  # Assume first two columns are key and value
        for row in reader:
            key = row[key_col]
            value = row[value_col]
            result[key] = float(value) if '.' in value or 'e' in value.lower() else int(value)
    return result


