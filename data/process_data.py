from sqlalchemy import create_engine
import pandas as pd

import sys


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """Load and merge Figure Eight messages and categories

    Args:
        messages_filepath (str): message data CSV filepath
        categories_filepath (str): categories data CSV filepath

    Returns:
        pd.DataFrame: Merged message and categories dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on="id")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and map categories data to multiclass

    Args:
        df (pd.DataFrame): Merged message and categories dataframe

    Returns:
        pd.DataFrame: Merged message and categories dataframe with multiclass categories
    """
    categories = df["categories"].str.split(";", expand=True)

    column_names = categories.loc[0].str.split("-")

    category_colnames = []
    for column_name in column_names:
        category_colnames.append(column_name[0])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df = df.drop("categories", axis=1)
    df = pd.concat([df, categories], join="inner", axis=1)

    return df.drop_duplicates()


def save_data(df: pd.DataFrame, database_filepath: str):
    """Save preprocessed dataset

    Args:
        df (pd.DataFrame): Merged message and categories dataframe with multiclass categories
        database_filepath (str): filepath to database
    """

    engine = create_engine(f"sqlite:///{database_filepath}")

    database_filename = database_filepath.split("/")[-1]
    table_name = database_filename.split(".db")[0]

    df.to_sql(table_name, engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        cleaned_df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(cleaned_df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "disaster_response.db"
        )


if __name__ == "__main__":
    main()