# =============================================================================
# Auction Price Prediction - Enriched Data Utilities
# =============================================================================
"""
Utilities for loading and working with enriched demographic/economic datasets.

These datasets are based on Canadian Forward Sortation Areas (FSA) and provide
geographic context for auction price predictions.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import settings


# =============================================================================
# Data Loading
# =============================================================================


def load_population_data() -> pd.DataFrame:
    """
    Load Statistics Canada population and dwelling counts by FSA.

    Returns:
        DataFrame with columns:
        - GEO: Forward Sortation Area (str)
        - Population, 2021: Total population (int)
        - Total private dwellings, 2021: Total dwellings (int)
        - Private dwellings occupied by usual residents, 2021: Occupied dwellings (int)

    Raises:
        FileNotFoundError: If the CSV file is not found
    """
    data_path = settings.data_dir / "enriched" / "population_dwelling_2021_fsa.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Population data not found at {data_path}. "
            "Please download from Statistics Canada. "
            "See data/enriched/README.md for instructions."
        )

    df = pd.read_csv(data_path)

    # Clean column names (remove trailing spaces)
    df.columns = df.columns.str.strip()

    # Ensure GEO is uppercase
    df["GEO"] = df["GEO"].str.upper()

    return df


def load_tax_statistics() -> pd.DataFrame:
    """
    Load CRA individual tax statistics by FSA.

    Returns:
        DataFrame with columns:
        - FSA: Forward Sortation Area (str)
        - Number of Returns: Total tax returns (int)
        - Total Income: Aggregate income in dollars (int)
        - Median Total Income: Median income in dollars (int)
        - Average Total Income: Average income in dollars (int)

    Raises:
        FileNotFoundError: If the CSV file is not found
    """
    data_path = settings.data_dir / "enriched" / "tax_stats_2021_fsa.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Tax statistics not found at {data_path}. "
            "Please download from CRA. "
            "See data/enriched/README.md for instructions."
        )

    df = pd.read_csv(data_path)

    # Ensure FSA is uppercase
    df["FSA"] = df["FSA"].str.upper()

    return df


def load_all_enriched_data() -> pd.DataFrame:
    """
    Load and merge all enriched datasets into a single DataFrame.

    Returns:
        Merged DataFrame indexed by FSA with all enriched features.
        Columns include population, dwelling, and tax statistics.

    Example:
        >>> enriched = load_all_enriched_data()
        >>> enriched.loc['M5V']  # Get data for Toronto FSA
    """
    population = load_population_data()
    tax_stats = load_tax_statistics()

    # Merge on FSA
    merged = population.merge(
        tax_stats, left_on="GEO", right_on="FSA", how="outer"
    )

    # Use GEO as the canonical FSA column
    merged["FSA"] = merged["GEO"].fillna(merged["FSA"])
    merged = merged.drop(columns=["GEO"])

    # Set FSA as index
    merged = merged.set_index("FSA")

    return merged


# =============================================================================
# FSA Extraction
# =============================================================================


def extract_fsa(postal_code: Optional[str]) -> Optional[str]:
    """
    Extract Forward Sortation Area (first 3 characters) from postal code.

    Args:
        postal_code: Canadian postal code (e.g., "M5V 3A8", "M5V3A8")

    Returns:
        FSA (e.g., "M5V") or None if postal code is invalid/missing

    Examples:
        >>> extract_fsa("M5V 3A8")
        'M5V'
        >>> extract_fsa("M5V3A8")
        'M5V'
        >>> extract_fsa("m5v 3a8")
        'M5V'
        >>> extract_fsa(None)
        None
    """
    if not postal_code or not isinstance(postal_code, str):
        return None

    # Remove spaces and take first 3 characters
    fsa = postal_code.replace(" ", "")[:3].upper()

    # Validate format (letter-digit-letter)
    if len(fsa) == 3 and fsa[0].isalpha() and fsa[1].isdigit() and fsa[2].isalpha():
        return fsa

    return None


def add_fsa_column(df: pd.DataFrame, postal_code_col: str = "postal_code") -> pd.DataFrame:
    """
    Add FSA column to DataFrame by extracting from postal codes.

    Args:
        df: DataFrame with postal code column
        postal_code_col: Name of column containing postal codes

    Returns:
        DataFrame with new 'fsa' column

    Example:
        >>> auctions = pd.DataFrame({'postal_code': ['M5V 3A8', 'K1A 0B1']})
        >>> auctions = add_fsa_column(auctions)
        >>> auctions['fsa'].tolist()
        ['M5V', 'K1A']
    """
    df = df.copy()
    df["fsa"] = df[postal_code_col].apply(extract_fsa)
    return df


# =============================================================================
# Enrichment
# =============================================================================


def enrich_with_demographics(
    df: pd.DataFrame,
    postal_code_col: str = "postal_code",
    fsa_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Enrich DataFrame with demographic and economic data by FSA.

    Args:
        df: DataFrame to enrich (must have postal codes or FSA)
        postal_code_col: Name of postal code column (if FSA not provided)
        fsa_col: Name of FSA column (if already extracted)

    Returns:
        DataFrame with enriched columns added

    Example:
        >>> auctions = load_auctions()
        >>> auctions_enriched = enrich_with_demographics(auctions)
        >>> auctions_enriched[['auction_id', 'fsa', 'median_income']]
    """
    df = df.copy()

    # Extract FSA if not provided
    if fsa_col is None:
        df = add_fsa_column(df, postal_code_col)
        fsa_col = "fsa"

    # Load enriched data
    enriched = load_all_enriched_data()

    # Merge
    df_enriched = df.merge(enriched, left_on=fsa_col, right_index=True, how="left")

    return df_enriched


# =============================================================================
# Feature Engineering
# =============================================================================


def compute_enriched_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from enriched demographic/economic data.

    Assumes DataFrame has enriched columns from load_all_enriched_data().

    Args:
        df: DataFrame with enriched columns

    Returns:
        DataFrame with additional derived feature columns

    Derived Features:
        - persons_per_dwelling: Population / occupied dwellings (urbanization proxy)
        - occupancy_rate: Occupied dwellings / total dwellings
        - income_skew: (Average - Median) / Median income
        - returns_per_capita: Tax returns / population
        - economic_capacity: Median income * population (in billions)
    """
    df = df.copy()

    # Persons per dwelling (urbanization proxy)
    if "Population, 2021" in df.columns and "Private dwellings occupied by usual residents, 2021" in df.columns:
        df["persons_per_dwelling"] = (
            df["Population, 2021"] / df["Private dwellings occupied by usual residents, 2021"]
        )

    # Occupancy rate
    if "Private dwellings occupied by usual residents, 2021" in df.columns and "Total private dwellings, 2021" in df.columns:
        df["occupancy_rate"] = (
            df["Private dwellings occupied by usual residents, 2021"] / df["Total private dwellings, 2021"]
        )

    # Income skew
    if "Average Total Income" in df.columns and "Median Total Income" in df.columns:
        df["income_skew"] = (
            (df["Average Total Income"] - df["Median Total Income"]) / df["Median Total Income"]
        )

    # Returns per capita
    if "Number of Returns" in df.columns and "Population, 2021" in df.columns:
        df["returns_per_capita"] = df["Number of Returns"] / df["Population, 2021"]

    # Economic capacity (in billions)
    if "Median Total Income" in df.columns and "Population, 2021" in df.columns:
        df["economic_capacity"] = (df["Median Total Income"] * df["Population, 2021"]) / 1e9

    return df


# =============================================================================
# Convenience Function
# =============================================================================


def enrich_auctions(
    df: pd.DataFrame,
    postal_code_col: str = "postal_code",
    compute_features: bool = True,
) -> pd.DataFrame:
    """
    One-step enrichment: add demographic data and compute derived features.

    Args:
        df: Auction DataFrame with postal codes
        postal_code_col: Name of postal code column
        compute_features: Whether to compute derived features

    Returns:
        Enriched DataFrame with demographic data and (optionally) derived features

    Example:
        >>> auctions = load_auctions()
        >>> auctions = enrich_auctions(auctions)
        >>> print(auctions[['auction_id', 'median_income', 'density_proxy']])
    """
    df_enriched = enrich_with_demographics(df, postal_code_col=postal_code_col)

    if compute_features:
        df_enriched = compute_enriched_features(df_enriched)

    return df_enriched
