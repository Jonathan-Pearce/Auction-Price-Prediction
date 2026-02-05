# =============================================================================
# Bid Feature Extraction
# =============================================================================
"""
Feature extraction from bid data for tabular ML models.

Provides utilities to aggregate bids per item into summary statistics
and extract time-based, distribution, and proxy bid features.
"""

from typing import Any

import pandas as pd
from loguru import logger

from src.feature_engineering.feature_config import FeatureConfig, load_feature_config


class BidFeatureExtractor:
    """
    Extract features from bid data for tabular ML models.

    This class aggregates bids per item into summary statistics including:
    - Bid amount statistics (max, min, mean, median, std, range)
    - Bid count features (total bids, unique bidders proxy)
    - Time-based features (duration, time gaps)
    - Distribution features (bid concentration)
    - Proxy bid features

    Args:
        config: FeatureConfig instance. If None, loads from default config.

    Example:
        >>> extractor = BidFeatureExtractor()
        >>> features_df = extractor.extract_features(bid_df)
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or load_feature_config()
        self._feature_names: list[str] = []

    @property
    def feature_config(self) -> Any:
        """Get features configuration."""
        return self.config.features

    def extract_features(self, bid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all configured features from bid data.

        Args:
            bid_df: DataFrame with bid data. Expected columns:
                - auction_id: int
                - item_id: int
                - bid_time: datetime string
                - bid_amount: float
                - bid_is_proxy: bool
                - bid_count: int (optional, total bids for item)

        Returns:
            DataFrame with one row per item and extracted features.
        """
        logger.info("Extracting features from bid data...")

        # Ensure bid_time is datetime
        bid_df = bid_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(bid_df["bid_time"]):
            # Use utc=True to handle mixed timezone data, then remove timezone info
            bid_df["bid_time"] = pd.to_datetime(bid_df["bid_time"], utc=True)
            # Convert to timezone-naive for consistent time calculations
            bid_df["bid_time"] = bid_df["bid_time"].dt.tz_localize(None)

        # Sort by item and time
        bid_df = bid_df.sort_values(["auction_id", "item_id", "bid_time"])

        # Group by item
        grouped = bid_df.groupby(["auction_id", "item_id"])

        # Initialize results
        all_features = []

        for (auction_id, item_id), group in grouped:
            item_features = {
                "auction_id": auction_id,
                "item_id": item_id,
            }

            # Extract each feature group
            if self.feature_config.bid_amount.enabled:
                item_features.update(self._extract_bid_amount_features(group))

            if self.feature_config.bid_count.enabled:
                item_features.update(self._extract_bid_count_features(group))

            if self.feature_config.time_features.enabled:
                item_features.update(self._extract_time_features(group))

            if self.feature_config.distribution_features.enabled:
                item_features.update(self._extract_distribution_features(group))

            if self.feature_config.proxy_features.enabled:
                item_features.update(self._extract_proxy_features(group))

            if self.feature_config.velocity_features.enabled:
                item_features.update(self._extract_velocity_features(group))

            all_features.append(item_features)

        result_df = pd.DataFrame(all_features)
        self._feature_names = [
            col for col in result_df.columns if col not in ["auction_id", "item_id"]
        ]

        logger.info(
            f"Extracted {len(self._feature_names)} features for {len(result_df):,} items"
        )
        return result_df

    def _extract_bid_amount_features(self, group: pd.DataFrame) -> dict[str, float]:
        """Extract bid amount statistics."""
        features = {}
        amounts = group["bid_amount"]

        # Get target (winning price)
        target_agg = self.feature_config.target.aggregation
        if target_agg == "max":
            features["winning_price"] = amounts.max()
        elif target_agg == "last":
            features["winning_price"] = amounts.iloc[-1]
        else:
            features["winning_price"] = amounts.max()

        # Statistics
        stats_config = self.feature_config.bid_amount.statistics
        if "max" in stats_config:
            features["bid_amount_max"] = amounts.max()
        if "min" in stats_config:
            features["bid_amount_min"] = amounts.min()
        if "mean" in stats_config:
            features["bid_amount_mean"] = amounts.mean()
        if "median" in stats_config:
            features["bid_amount_median"] = amounts.median()
        if "std" in stats_config:
            features["bid_amount_std"] = amounts.std() if len(amounts) > 1 else 0.0

        # Derived features
        derived_config = self.feature_config.bid_amount.derived
        if "range" in derived_config:
            features["bid_amount_range"] = amounts.max() - amounts.min()
        if "coefficient_of_variation" in derived_config:
            mean_val = amounts.mean()
            std_val = amounts.std() if len(amounts) > 1 else 0.0
            features["bid_amount_cv"] = std_val / mean_val if mean_val > 0 else 0.0

        return features

    def _extract_bid_count_features(self, group: pd.DataFrame) -> dict[str, Any]:
        """Extract bid count features."""
        features = {}
        features_config = self.feature_config.bid_count.features

        if "total_bids" in features_config:
            features["total_bids"] = len(group)

        if "unique_bidders_proxy" in features_config:
            # Approximate unique bidders by counting transitions between proxy/non-proxy
            # This is a heuristic since we don't have actual bidder IDs
            if "bid_is_proxy" in group.columns:
                # Count number of consecutive bid type changes as proxy for different bidders
                proxy_changes = (
                    group["bid_is_proxy"] != group["bid_is_proxy"].shift()
                ).sum()
                features["unique_bidders_proxy"] = max(1, proxy_changes)
            else:
                features["unique_bidders_proxy"] = 1

        return features

    def _extract_time_features(self, group: pd.DataFrame) -> dict[str, float]:
        """Extract time-based features."""
        features = {}
        times = group["bid_time"]
        features_config = self.feature_config.time_features.features

        if len(times) < 2:
            # Not enough bids for time features
            for feat in features_config:
                if feat in ["bidding_duration_seconds", "first_last_bid_delta"]:
                    features[feat] = 0.0
                elif feat in ["mean_time_between_bids", "median_time_between_bids"]:
                    features[feat] = 0.0
            return features

        # Convert to seconds since first bid
        first_time = times.iloc[0]
        last_time = times.iloc[-1]
        duration = (last_time - first_time).total_seconds()

        if "bidding_duration_seconds" in features_config:
            features["bidding_duration_seconds"] = duration

        if "first_last_bid_delta" in features_config:
            features["first_last_bid_delta"] = duration

        # Time gaps between consecutive bids
        time_diffs = times.diff().dropna()
        time_gaps_seconds = time_diffs.dt.total_seconds()

        if "mean_time_between_bids" in features_config:
            features["mean_time_between_bids"] = (
                time_gaps_seconds.mean() if len(time_gaps_seconds) > 0 else 0.0
            )

        if "median_time_between_bids" in features_config:
            features["median_time_between_bids"] = (
                time_gaps_seconds.median() if len(time_gaps_seconds) > 0 else 0.0
            )

        return features

    def _extract_distribution_features(self, group: pd.DataFrame) -> dict[str, float]:
        """Extract distribution/concentration features."""
        features = {}
        dist_config = self.feature_config.distribution_features

        times = group["bid_time"]
        amounts = group["bid_amount"]

        if len(times) < 2:
            # Not enough bids for distribution features
            for feat in dist_config.features:
                features[feat] = 0.0
            if dist_config.last_n_bids.enabled:
                for n in dist_config.last_n_bids.n_values:
                    for feat in dist_config.last_n_bids.features:
                        features[f"last_{n}_bids_{feat}"] = 0.0
            return features

        # Bid concentration features
        first_time = times.iloc[0]
        last_time = times.iloc[-1]
        duration = (last_time - first_time).total_seconds()

        if duration > 0:
            # Fraction of bids in last 25% of duration
            if "bid_concentration_last_25pct" in dist_config.features:
                threshold_25 = first_time + pd.Timedelta(seconds=duration * 0.75)
                bids_in_last_25 = (times >= threshold_25).sum()
                features["bid_concentration_last_25pct"] = bids_in_last_25 / len(times)

            # Fraction of bids in last 10% of duration
            if "bid_concentration_last_10pct" in dist_config.features:
                threshold_10 = first_time + pd.Timedelta(seconds=duration * 0.90)
                bids_in_last_10 = (times >= threshold_10).sum()
                features["bid_concentration_last_10pct"] = bids_in_last_10 / len(times)
        else:
            if "bid_concentration_last_25pct" in dist_config.features:
                features["bid_concentration_last_25pct"] = 1.0
            if "bid_concentration_last_10pct" in dist_config.features:
                features["bid_concentration_last_10pct"] = 1.0

        # Last N bids features
        if dist_config.last_n_bids.enabled:
            for n in dist_config.last_n_bids.n_values:
                last_n_amounts = amounts.tail(n)

                for feat in dist_config.last_n_bids.features:
                    if feat == "mean_amount":
                        features[f"last_{n}_bids_mean_amount"] = last_n_amounts.mean()
                    elif feat == "amount_growth_rate":
                        if len(last_n_amounts) > 1:
                            # Average increment between consecutive bids
                            increments = last_n_amounts.diff().dropna()
                            features[f"last_{n}_bids_amount_growth_rate"] = (
                                increments.mean()
                            )
                        else:
                            features[f"last_{n}_bids_amount_growth_rate"] = 0.0

        return features

    def _extract_proxy_features(self, group: pd.DataFrame) -> dict[str, float]:
        """Extract proxy bid features."""
        features = {}
        proxy_config = self.feature_config.proxy_features

        if "bid_is_proxy" not in group.columns:
            for feat in proxy_config.features:
                features[feat] = 0.0
            return features

        proxy_bids = group["bid_is_proxy"]

        if "proxy_bid_count" in proxy_config.features:
            features["proxy_bid_count"] = proxy_bids.sum()

        if "proxy_bid_ratio" in proxy_config.features:
            features["proxy_bid_ratio"] = (
                proxy_bids.sum() / len(proxy_bids) if len(proxy_bids) > 0 else 0.0
            )

        return features

    def _extract_velocity_features(self, group: pd.DataFrame) -> dict[str, float]:
        """Extract velocity and acceleration features."""
        features = {}
        velocity_config = self.feature_config.velocity_features

        times = group["bid_time"]
        amounts = group["bid_amount"]

        if len(times) < 2:
            # Not enough bids for velocity features
            for feat in velocity_config.features:
                features[feat] = 0.0
            return features

        # Calculate duration in minutes
        first_time = times.iloc[0]
        last_time = times.iloc[-1]
        duration_seconds = (last_time - first_time).total_seconds()
        duration_minutes = duration_seconds / 60.0

        # Bid velocity: bids per minute
        if "bid_velocity" in velocity_config.features:
            features["bid_velocity"] = (
                len(times) / duration_minutes if duration_minutes > 0 else 0.0
            )

        # Bid amount velocity: average $ increase per minute
        if "bid_amount_velocity" in velocity_config.features:
            total_increase = amounts.iloc[-1] - amounts.iloc[0]
            features["bid_amount_velocity"] = (
                total_increase / duration_minutes if duration_minutes > 0 else 0.0
            )

        # Calculate time differences for acceleration
        time_diffs = times.diff().dropna()
        time_diffs_minutes = time_diffs.dt.total_seconds() / 60.0

        # Bid acceleration: rate of change of velocity
        if "bid_acceleration" in velocity_config.features:
            if len(time_diffs_minutes) >= 2:
                # Calculate instantaneous velocities (1/time_gap)
                # Avoid division by zero
                velocities = []
                for td in time_diffs_minutes:
                    if td > 0:
                        velocities.append(1.0 / td)
                    else:
                        velocities.append(0.0)

                if len(velocities) >= 2:
                    # Acceleration is change in velocity
                    velocity_changes = [
                        velocities[i] - velocities[i - 1]
                        for i in range(1, len(velocities))
                    ]
                    features["bid_acceleration"] = (
                        sum(velocity_changes) / len(velocity_changes)
                        if velocity_changes
                        else 0.0
                    )
                else:
                    features["bid_acceleration"] = 0.0
            else:
                features["bid_acceleration"] = 0.0

        # Bid amount acceleration: rate of change of amount velocity
        if "bid_amount_acceleration" in velocity_config.features:
            amount_diffs = amounts.diff().dropna()
            if len(amount_diffs) >= 2 and len(time_diffs_minutes) >= 2:
                # Calculate instantaneous amount velocities
                amount_velocities = []
                for amt_diff, td in zip(amount_diffs, time_diffs_minutes, strict=False):
                    if td > 0:
                        amount_velocities.append(amt_diff / td)
                    else:
                        amount_velocities.append(0.0)

                if len(amount_velocities) >= 2:
                    velocity_changes = [
                        amount_velocities[i] - amount_velocities[i - 1]
                        for i in range(1, len(amount_velocities))
                    ]
                    features["bid_amount_acceleration"] = (
                        sum(velocity_changes) / len(velocity_changes)
                        if velocity_changes
                        else 0.0
                    )
                else:
                    features["bid_amount_acceleration"] = 0.0
            else:
                features["bid_amount_acceleration"] = 0.0

        # Maximum bid velocity: max bids per minute in any time window
        if "max_bid_velocity" in velocity_config.features:
            if len(time_diffs_minutes) > 0:
                # Calculate velocity for each gap (1/gap gives bids per minute)
                instant_velocities = []
                for td in time_diffs_minutes:
                    if td > 0:
                        instant_velocities.append(1.0 / td)
                features["max_bid_velocity"] = (
                    max(instant_velocities) if instant_velocities else 0.0
                )
            else:
                features["max_bid_velocity"] = 0.0

        # Final bid velocity: bid velocity in last 25% of auction
        if "final_bid_velocity" in velocity_config.features:
            if duration_seconds > 0:
                threshold_time = first_time + pd.Timedelta(
                    seconds=duration_seconds * 0.75
                )
                final_bids = times[times >= threshold_time]
                final_duration_minutes = duration_seconds * 0.25 / 60.0
                features["final_bid_velocity"] = (
                    len(final_bids) / final_duration_minutes
                    if final_duration_minutes > 0
                    else 0.0
                )
            else:
                features["final_bid_velocity"] = 0.0

        return features

    def get_feature_names(self) -> list[str]:
        """
        Get list of feature names (excluding identifiers).

        Returns:
            List of feature column names.
        """
        return self._feature_names

    def get_feature_metadata(self) -> dict[str, dict[str, str]]:
        """
        Get metadata about each feature.

        Returns:
            Dictionary mapping feature names to their descriptions and types.
        """
        metadata = {
            "winning_price": {
                "description": "Target variable: winning (maximum) bid amount",
                "type": "target",
            },
            "bid_amount_max": {
                "description": "Maximum bid amount (same as winning price)",
                "type": "bid_amount",
            },
            "bid_amount_min": {
                "description": "Minimum bid amount",
                "type": "bid_amount",
            },
            "bid_amount_mean": {
                "description": "Mean bid amount",
                "type": "bid_amount",
            },
            "bid_amount_median": {
                "description": "Median bid amount",
                "type": "bid_amount",
            },
            "bid_amount_std": {
                "description": "Standard deviation of bid amounts",
                "type": "bid_amount",
            },
            "bid_amount_range": {
                "description": "Range of bid amounts (max - min)",
                "type": "bid_amount",
            },
            "bid_amount_cv": {
                "description": "Coefficient of variation (std / mean)",
                "type": "bid_amount",
            },
            "total_bids": {
                "description": "Total number of bids on the item",
                "type": "bid_count",
            },
            "unique_bidders_proxy": {
                "description": "Estimated unique bidders (proxy based on bid patterns)",
                "type": "bid_count",
            },
            "bidding_duration_seconds": {
                "description": "Time between first and last bid in seconds",
                "type": "time",
            },
            "first_last_bid_delta": {
                "description": "Alias for bidding_duration_seconds",
                "type": "time",
            },
            "mean_time_between_bids": {
                "description": "Average time gap between consecutive bids (seconds)",
                "type": "time",
            },
            "median_time_between_bids": {
                "description": "Median time gap between consecutive bids (seconds)",
                "type": "time",
            },
            "bid_concentration_last_25pct": {
                "description": "Fraction of bids in the last 25% of bidding duration",
                "type": "distribution",
            },
            "bid_concentration_last_10pct": {
                "description": "Fraction of bids in the last 10% of bidding duration",
                "type": "distribution",
            },
            "proxy_bid_count": {
                "description": "Number of proxy (automated) bids",
                "type": "proxy",
            },
            "proxy_bid_ratio": {
                "description": "Fraction of bids that are proxy bids",
                "type": "proxy",
            },
        }

        # Add last N bids features
        if self.feature_config.distribution_features.last_n_bids.enabled:
            for n in self.feature_config.distribution_features.last_n_bids.n_values:
                metadata[f"last_{n}_bids_mean_amount"] = {
                    "description": f"Mean bid amount of last {n} bids",
                    "type": "distribution",
                }
                metadata[f"last_{n}_bids_amount_growth_rate"] = {
                    "description": f"Average bid increment in last {n} bids",
                    "type": "distribution",
                }

        # Add velocity features
        metadata["bid_velocity"] = {
            "description": "Bids per minute (overall)",
            "type": "velocity",
        }
        metadata["bid_amount_velocity"] = {
            "description": "Average dollar increase per minute",
            "type": "velocity",
        }
        metadata["bid_acceleration"] = {
            "description": "Rate of change of bid velocity (bids/min²)",
            "type": "velocity",
        }
        metadata["bid_amount_acceleration"] = {
            "description": "Rate of change of amount velocity ($/min²)",
            "type": "velocity",
        }
        metadata["max_bid_velocity"] = {
            "description": "Maximum instantaneous bid velocity",
            "type": "velocity",
        }
        metadata["final_bid_velocity"] = {
            "description": "Bid velocity in last 25% of auction duration",
            "type": "velocity",
        }

        return metadata


def aggregate_bids_per_item(
    bid_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """
    Convenience function to aggregate bids per item into features.

    Args:
        bid_df: DataFrame with bid data.
        config: FeatureConfig instance. If None, loads from default config.

    Returns:
        DataFrame with one row per item and extracted features.

    Example:
        >>> from src.feature_engineering import load_bid_data, aggregate_bids_per_item
        >>> bid_df = load_bid_data(limit=10000)
        >>> features_df = aggregate_bids_per_item(bid_df)
        >>> print(features_df.columns)
    """
    extractor = BidFeatureExtractor(config)
    return extractor.extract_features(bid_df)
