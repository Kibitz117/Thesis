import torch
import os

def save_train_test_splits(input_file, output_dir):
    """
    Loads train/test splits from a specified file and saves each period split into a separate file.

    Parameters:
    - input_file: The path to the file containing the train/test splits.
    - output_dir: The directory where the split files should be saved.
    """

    # Load the data
    data = torch.load(input_file)
    train_test_splits = data['train_test_splits']

    # Create the output directory if it doesn't exist    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each period and save it as a separate file
    for i, period_split in enumerate(train_test_splits):
        file_path = os.path.join(output_dir, f'period_{i}.pt')
        torch.save(period_split, file_path)
        print(f'Saved {file_path}')

# Example usage
