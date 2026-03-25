import csv

def save_dict_to_csv(data_dict: dict, filename: str):
    """
    Saves data from a dictionary of lists to a CSV file.

    The keys of the dictionary are used as column headers. This function can
    handle an arbitrary number of columns.

    Args:
        data_dict (dict): A dictionary where keys are the column headers (str)
                          and values are the lists of data for that column.
                          All lists must be of the same length.
        filename (str): The name of the output CSV file.
    """
    if not data_dict:
        print("Warning: No data provided to save. CSV file will not be created.")
        return

    # Check if all lists have the same length by comparing them to the first one.
    try:
        it = iter(data_dict.values())
        length = len(next(it))
        if not all(len(l) == length for l in it):
            # If lengths are inconsistent, raise an error.
            raise ValueError("All lists in the data dictionary must have the same length.")
    except StopIteration:
        # This handles the case of an empty dictionary.
        print("Warning: Data dictionary is empty. CSV file will not be created.")
        return


    header = list(data_dict.keys())
    # Transpose the data from columns (dictionary values) to rows
    rows = zip(*data_dict.values())

    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(header)
            # Write all the data rows
            writer.writerows(rows)
        print(f"Successfully saved data to {filename}")
    except IOError as e:
        print(f"Error: Unable to write to file {filename}. Reason: {e}")

