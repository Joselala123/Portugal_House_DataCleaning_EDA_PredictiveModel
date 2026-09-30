import pandas as pd
from IPython.display import display


def display_base_columns(df):
    """
    Displays only the original dataset columns.
    """
    display(df[['Price', 'District', 'City', 'Town', 'EnergyCertificate', 'Floor',
       'NumberOfBathrooms', 'Parking', 'TotalArea', 'ConstructionYear',
       'EnergyEfficiencyLevel', 'PublishDate', 'Garage', 'Elevator',
       'ElectricCarsCharging', 'TotalRooms', 'NumberOfBedrooms', 'NumberOfWC',
       'ConservationStatus', 'LivingArea', 'LotSize', 'BuiltArea']])


def show_nulls(df):
    """
    Displays the number and percentage of missing values
    for each variable.
    """
    nulls = (
        pd.DataFrame({
            "nMissing": df.isna().sum(),
            "percentage": (df.isna().sum() / len(df)) * 100
        })
        .sort_values("percentage", ascending=False)
    )

    print("Count of null values for each variable and its percentage:")
    print()
    display(nulls)


def create_groups(
    df,
    group_columns,
    group_col,
    count_col,
    start_group_at_one=True
):
    """
    Groups observations using the specified columns.

    Creates a unique group ID and counts how many
    observations belong to each group.
    """

    df = df.copy()

    groups = df.groupby(
        group_columns,
        dropna=False
    )

    group_ids = groups.ngroup()
    group_counts = groups[group_columns[0]].transform("size")

    if start_group_at_one:
        group_ids += 1

    df[group_col] = group_ids
    df[count_col] = group_counts

    return df


def assign_duplicate_confidence(
    df,
    condition,
    confidence_level
):
    """
    Assigns a confidence level to observations that
    satisfy the specified condition.
    """

    df.loc[
        condition,
        "DuplicateConfidence"
    ] = confidence_level

    return df
