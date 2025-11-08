"""Data loading and preprocessing utilities for algorithmic progress analysis."""

import pandas as pd
import numpy as np


def load_eci_from_website(csv_path="data/website/epoch_capabilities_index.csv"):
    """Load ECI scores from website data.

    Args:
        csv_path: Path to website CSV file

    Returns:
        DataFrame with columns: model, date, date_obj, estimated_capability, compute, Model
    """
    print(f"Loading data from {csv_path}...")
    eci_df = pd.read_csv(csv_path)

    # Rename columns to match expected format
    column_mapping = {
        'Model version': 'model',
        'ECI Score': 'estimated_capability',
        'Release date': 'date',
        'Training compute (FLOP)': 'compute'
    }
    eci_df = eci_df.rename(columns=column_mapping)

    # Parse date and create Model column for consistency
    eci_df['date_obj'] = pd.to_datetime(eci_df['date'])
    eci_df['Model'] = eci_df['model']

    print(f"Loaded {len(eci_df)} models from website data")
    return eci_df


def load_eci_from_outputs(csv_path="outputs/model_fit/model_capabilities.csv"):
    """Load ECI scores from model fit outputs.

    Args:
        csv_path: Path to model capabilities CSV file

    Returns:
        DataFrame with columns: model, date, date_obj, estimated_capability, Model
    """
    print(f"Loading ECI scores from {csv_path}...")
    eci_df = pd.read_csv(csv_path)
    print(f"Loaded ECI scores for {len(eci_df)} models")

    # Ensure date_obj is datetime
    if 'date_obj' not in eci_df.columns:
        eci_df['date_obj'] = pd.to_datetime(eci_df['date'])
    else:
        eci_df['date_obj'] = pd.to_datetime(eci_df['date_obj'])

    return eci_df


def filter_distilled_models(df, exclude_distilled=False, exclude_med_high_distilled=False,
                            distilled_csv="data/distilled_models.csv"):
    """Filter out distilled models based on confidence levels.

    Args:
        df: DataFrame with 'model' column
        exclude_distilled: If True, exclude all distilled models (all confidence levels)
        exclude_med_high_distilled: If True, exclude med/high confidence distilled models
        distilled_csv: Path to distilled models CSV

    Returns:
        Filtered DataFrame
    """
    if not exclude_distilled and not exclude_med_high_distilled:
        return df

    print("\nFiltering out distilled models...")
    distilled_df = pd.read_csv(distilled_csv)

    # Determine which confidence levels to exclude
    if exclude_distilled:
        confidence_levels = ['high', 'medium', 'low']
        print("  Excluding: ALL distilled models (high, medium, and low confidence)")
    elif exclude_med_high_distilled:
        confidence_levels = ['high', 'medium']
        print("  Excluding: medium and high confidence distilled models only")

    distilled_models = distilled_df[
        (distilled_df['distilled'] == True) &
        (distilled_df['confidence'].isin(confidence_levels))
    ]['model'].tolist()

    before_count = len(df)
    df = df[~df['model'].isin(distilled_models)].copy()
    after_count = len(df)

    print(f"Excluded {before_count - after_count} distilled models "
          f"({100 * (before_count - after_count) / before_count:.1f}%)")
    print(f"Remaining models: {after_count}")

    return df


def filter_by_release_date(df, min_release_date=None):
    """Filter models by release date.

    Args:
        df: DataFrame with 'date_obj' column
        min_release_date: Minimum release date (string in YYYY-MM-DD format or datetime)

    Returns:
        Filtered DataFrame
    """
    if min_release_date is None:
        return df

    print(f"\nFiltering models released on or after {min_release_date}...")
    min_date_obj = pd.to_datetime(min_release_date)

    before_count = len(df)
    df = df[df['date_obj'] >= min_date_obj].copy()
    after_count = len(df)

    print(f"Excluded {before_count - after_count} models released before {min_release_date}")
    print(f"Remaining models: {after_count}")

    return df


def merge_compute_data(df, compute_csv="data/all_ai_models.csv"):
    """Merge compute data from PCD dataset.

    Args:
        df: DataFrame with 'Model' column
        compute_csv: Path to compute data CSV

    Returns:
        DataFrame with additional columns: compute, parameters, data
    """
    try:
        # Read full dataset
        pcd_dataset = pd.read_csv(compute_csv)

        # Check which dataset size column exists
        dataset_col = None
        for col in ["Training dataset size (datapoints)", "Training dataset size (gradients)"]:
            if col in pcd_dataset.columns:
                dataset_col = col
                break

        # Select columns that exist
        cols_to_keep = ["Model", "Training compute (FLOP)", "Parameters"]
        if dataset_col:
            cols_to_keep.append(dataset_col)

        pcd_dataset = pcd_dataset[cols_to_keep]

        # Rename columns
        columns = {
            "Training compute (FLOP)": "compute",
            "Parameters": "parameters"
        }
        if dataset_col:
            columns[dataset_col] = "data"

        pcd_dataset = pcd_dataset.rename(columns=columns)

        df = df.merge(pcd_dataset, on="Model", how="left")
        print(f"Merged compute data: {df['compute'].notna().sum()} models have compute info")
        return df
    except Exception as e:
        print(f"Error: Could not load compute data: {e}")
        return None


def prepare_for_analysis(df, reference_date='2020-01-01'):
    """Add derived columns needed for analysis.

    Args:
        df: DataFrame with date_obj and compute columns
        reference_date: Reference date for numeric date conversion

    Returns:
        DataFrame with additional columns: log_compute, date_numeric
    """
    df = df.copy()

    # Add log compute
    if 'compute' in df.columns:
        df['log_compute'] = np.log10(df['compute'])

    # Convert date to numeric (years since reference)
    df['date_numeric'] = (df['date_obj'] -
                          pd.Timestamp(reference_date)).dt.total_seconds() / (365.25 * 24 * 3600)

    return df


def load_model_capabilities_and_compute(use_website_data=False,
                                       exclude_distilled=False,
                                       exclude_med_high_distilled=False,
                                       filter_complete=True,
                                       min_release_date=None):
    """Load ECI scores and merge with compute data.

    This is the main entry point for loading data for algorithmic progress analysis.

    Args:
        use_website_data: If True, load from data/website/epoch_capabilities_index.csv
        exclude_distilled: If True, exclude all distilled models (all confidence levels)
        exclude_med_high_distilled: If True, exclude med/high confidence distilled models
        filter_complete: If True, only return models with complete data
        min_release_date: If provided, only include models released on or after this date

    Returns:
        DataFrame with all necessary columns for analysis, or None on error
    """
    # Load ECI data
    if use_website_data:
        eci_df = load_eci_from_website()

        # Filter distilled models if requested
        eci_df = filter_distilled_models(eci_df, exclude_distilled, exclude_med_high_distilled)

        # Filter by release date if requested
        eci_df = filter_by_release_date(eci_df, min_release_date)

        # Verify anchor models if present
        anchor_models = eci_df[eci_df['model'].isin([
            'claude-3-5-sonnet-20240620', 'gpt-5-2025-08-07_medium'])]
        if len(anchor_models) > 0:
            print("\nAnchor models in website data:")
            for _, row in anchor_models.iterrows():
                print(f"  {row['model']}: ECI={row['estimated_capability']:.1f}")

        print(f"Compute data from website: {eci_df['compute'].notna().sum()} models have compute info")
        df = eci_df.copy()

    else:
        eci_df = load_eci_from_outputs()

        # Filter distilled models if requested
        eci_df = filter_distilled_models(eci_df, exclude_distilled, exclude_med_high_distilled)

        # Filter by release date if requested
        eci_df = filter_by_release_date(eci_df, min_release_date)

        # Verify anchor models
        anchor_models = eci_df[eci_df['model'].isin([
            'claude-3-5-sonnet-20240620', 'gpt-5-2025-08-07_medium'])]
        if len(anchor_models) > 0:
            print("\nAnchor models in ECI data:")
            for _, row in anchor_models.iterrows():
                print(f"  {row['model']}: ECI={row['estimated_capability']:.1f}")

        # Merge compute data
        df = merge_compute_data(eci_df)
        if df is None:
            return None

    # Prepare for analysis
    df = prepare_for_analysis(df)

    # Filter to complete data if requested
    if filter_complete:
        df = df.dropna(subset=['date_obj', 'compute', 'estimated_capability']).copy()
        print(f"Prepared {len(df)} models with complete data")

    return df
