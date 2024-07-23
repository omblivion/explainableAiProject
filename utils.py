def print_demographic_distribution(df):
    """
    Prints the distribution of gender and race in the given dataframe.

    :param df: The dataframe to analyze.
    """
    gender_counts = {
        "male": df["male"].sum(),
        "female": df["female"].sum(),
        "unknown": df["unknown"].sum()
    }

    race_counts = {
        "white": df["white"].sum(),
        "black": df["black"].sum(),
        "asian": df["asian"].sum(),
        "hispanic": df["hispanic"].sum(),
        "other": df["other"].sum(),
        "non-identified": df["non-identified"].sum()
    }

    print("Gender Distribution:")
    for gender, count in gender_counts.items():
        print(f"{gender.capitalize()}: {count}")

    print("\nRace Distribution:")
    for race, count in race_counts.items():
        print(f"{race.capitalize()}: {count}")
