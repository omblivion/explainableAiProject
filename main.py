if __name__ == "__main__":
    import argparse
    from dataset_load import DatasetLoad

    parser = argparse.ArgumentParser(description='Load dataset')
    parser.add_argument('--dataset_type', type=str, default='emotion', choices=['emotion', 'sarcasm'],
                        help='Type of dataset to load')
    args = parser.parse_args()

    dataset_loader = DatasetLoad(args.dataset_type)
    dataset_loader.load_datasets()

    if args.dataset_type == 'emotion':
        print("Train Data")
        print(dataset_loader.train_data.head())
        print("\nTest Data")
        print(dataset_loader.test_data.head())
        print("\nValidation Data")
        print(dataset_loader.val_data.head())
    elif args.dataset_type == 'sarcasm':
        print("Train Data")
        print(dataset_loader.train_data.head())
        print("\nValidation Data")
        print(dataset_loader.val_data.head())
        print("\nTest Data")
        print(dataset_loader.test_data.head())
