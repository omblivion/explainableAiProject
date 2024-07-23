import os

import pandas as pd


def augment_with_metadata_and_topic(dataset, extractor, file_path, debug=False):
    # Check if the augmented dataset already exists
    if os.path.exists(file_path):
        # If the file exists, load the augmented dataset from the CSV file
        print(f"Loading augmented dataset from {file_path}")
        augmented_dataset = pd.read_csv(file_path)
    else:
        # If the file does not exist, proceed with augmenting the dataset
        print(f"Augmenting dataset and saving to {file_path}")
        total_rows = len(dataset)  # Get the total number of rows in the dataset
        count = 0  # Initialize a counter to keep track of processed rows

        # Iterate over each row in the dataset
        for index, row in dataset.iterrows():
            # Extract gender using the extractor
            gender = extractor.extract_gender(row['text'])
            # Extract race using the extractor
            race = extractor.extract_race(row['text'])

            # Update the dataset with gender metadata
            for label in ["male", "female", "unknown"]:
                dataset.at[index, label] = 1 if gender == label else 0

            # Update the dataset with race metadata
            for label in ["white", "black", "asian", "hispanic", "other", "non-identified"]:
                dataset.at[index, label] = 1 if race == label else 0

            # If debug mode is enabled, print debug information
            if debug:
                # Calculate the percentage of completion
                percentage_complete = ((count + 1) / total_rows) * 100
                print(f"Text: {row['text']}")
                print(f"Generated Metadata: Gender - {gender}, Race - {race}")
                print(f"Percentage of Completion: {percentage_complete:.2f}%, {count + 1} of {total_rows}")

            # Increment the counter after processing each row
            count += 1

        # Save the augmented dataset to a CSV file
        dataset.to_csv(file_path, index=False)
        # Assign the augmented dataset to the return variable
        augmented_dataset = dataset

    # Return the augmented dataset
    return augmented_dataset
