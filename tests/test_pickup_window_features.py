# =============================================================================
# Tests for Pickup Window Feature Extraction
# =============================================================================
"""
Test suite for pickup window feature engineering functionality.

Tests cover:
- Basic single pickup window extraction
- Multiple category-based windows
- Edge cases (missing data, malformed HTML)
- Time format variations
"""

import pandas as pd

from src.features import engineer_auction_features, extract_pickup_windows


class TestExtractPickupWindows:
    """Test the extract_pickup_windows function."""
    
    def test_single_pickup_window_basic(self):
        """Test extraction of a single basic pickup window."""
        html = """
        <div class="bold">
            <span style="font-size: 12pt;">
                <strong>Pickup: Saturday, March 14 EDT, 9AM - 12 NOON</strong>
            </span>
        </div>
        """
        
        result = extract_pickup_windows(html)
        
        assert result["num_pickup_windows"] == 1
        assert result["total_pickup_hours"] == 3.0
        assert result["first_pickup_start_hour"] == 9.0
        assert result["last_pickup_end_hour"] == 12.0
        assert result["pickup_day_of_week"] == 5  # Saturday
        assert result["has_category_windows"] is False
    
    def test_single_pickup_window_pm(self):
        """Test extraction with PM times."""
        html = """
        <div class="bold">
            <span><strong>Pickup: Friday, March 13 EDT, 4PM - 7PM</strong></span>
        </div>
        """
        
        result = extract_pickup_windows(html)
        
        assert result["num_pickup_windows"] == 1
        assert result["total_pickup_hours"] == 3.0
        assert result["first_pickup_start_hour"] == 16.0  # 4PM = 16:00
        assert result["last_pickup_end_hour"] == 19.0  # 7PM = 19:00
        assert result["pickup_day_of_week"] == 4  # Friday
        assert result["has_category_windows"] is False
    
    def test_pickup_window_with_minutes(self):
        """Test extraction with time including minutes."""
        html = """
        <div class="bold">
            <strong>Pickup: Sunday, March 15 EDT, 12:00 Noon - 04:00 PM</strong>
        </div>
        """
        
        result = extract_pickup_windows(html)
        
        assert result["num_pickup_windows"] == 1
        assert result["total_pickup_hours"] == 4.0
        assert result["first_pickup_start_hour"] == 12.0
        assert result["last_pickup_end_hour"] == 16.0
        assert result["pickup_day_of_week"] == 6  # Sunday
    
    def test_pickup_window_with_categories(self):
        """Test extraction with category-based sub-windows."""
        html = """
        <div class="bold">
            <strong>Pickup: Wednesday, March 18 EDT, 4PM - 7PM</strong>
            <strong>Cat A: 4PM - 6PM</strong>
            <strong>Cat B: 6PM - 7PM</strong>
            <strong>Cat C: 6:30PM - 7PM</strong>
        </div>
        """
        
        result = extract_pickup_windows(html)
        
        # Should capture all time windows
        assert result["num_pickup_windows"] >= 3  # At least the category windows
        assert result["has_category_windows"] is True
        assert result["pickup_day_of_week"] == 2  # Wednesday
        assert result["first_pickup_start_hour"] == 16.0  # 4PM
        assert result["last_pickup_end_hour"] == 19.0  # 7PM
    
    def test_pickup_window_category_without_cat_prefix(self):
        """Test extraction with categories labeled differently."""
        html = """
        <div class="bold">
            <strong>Pickup: Saturday, March 14 EDT, 9AM - 12 NOON</strong>
            <strong>Category A: 9AM - 11AM</strong>
            <strong>Category B: 11AM - 12NOON</strong>
        </div>
        """
        
        result = extract_pickup_windows(html)
        
        assert result["num_pickup_windows"] >= 2
        assert result["has_category_windows"] is True
        assert result["pickup_day_of_week"] == 5  # Saturday
    
    def test_monday_pickup(self):
        """Test Monday pickup extraction."""
        html = """
        <strong>Pickup: Monday, March 23 EDT, 6:00PM - 7:30PM</strong>
        """
        
        result = extract_pickup_windows(html)
        
        assert result["pickup_day_of_week"] == 0  # Monday
        assert result["first_pickup_start_hour"] == 18.0
        assert result["last_pickup_end_hour"] == 19.5
    
    def test_tuesday_pickup(self):
        """Test Tuesday pickup extraction."""
        html = """
        <strong>Pickup: Tuesday, March 24 EDT, 10AM - 2PM</strong>
        """
        
        result = extract_pickup_windows(html)
        
        assert result["pickup_day_of_week"] == 1  # Tuesday
        assert result["total_pickup_hours"] == 4.0
    
    def test_empty_or_none_input(self):
        """Test handling of empty or None input."""
        result_none = extract_pickup_windows(None)
        result_empty = extract_pickup_windows("")
        
        assert result_none["num_pickup_windows"] == 0
        assert result_none["total_pickup_hours"] == 0.0
        assert result_none["first_pickup_start_hour"] is None
        
        assert result_empty["num_pickup_windows"] == 0
        assert result_empty["total_pickup_hours"] == 0.0
    
    def test_malformed_html(self):
        """Test handling of malformed HTML without crashing."""
        html = "<div>Some random text without pickup info</div>"
        
        result = extract_pickup_windows(html)
        
        # Should return defaults without crashing
        assert result["num_pickup_windows"] == 0
        assert result["total_pickup_hours"] == 0.0
    
    def test_real_example_1(self):
        """Test with real example from dataset."""
        html = """
        <div class="bold">
        <div class="bold">
        <div class="bold"><span style="font-size: 12pt;"><strong>Pickup: Friday, March 13 <span title="Eastern Standard Time, UTC -5 hours">EDT</span>, 4PM - 7PM</strong></span></div>
        <div class="bold"><span style="font-size: 12pt;"><strong>Pickup Location: 12 Earl St, Toronto, ON M4Y 1M3 <a href="https://www.google.com/maps/place/12+Earl+St,+Toronto,+ON+M4Y+1M3,+Canada/data=!4m2!3m1!1s0x89d4cb5263f52b0b:0x791352834898ed5b?sa=X&ei=UrvsVLa4PIPjuQS2kICADw&ved=0CB8Q8gEwAA" target="_blank" rel="nofollow">Map</a></strong></span></div>
        </div>
        <div class="bold"> </div>
        </div>
        """
        
        result = extract_pickup_windows(html)
        
        assert result["num_pickup_windows"] == 1
        assert result["total_pickup_hours"] == 3.0
        assert result["first_pickup_start_hour"] == 16.0
        assert result["last_pickup_end_hour"] == 19.0
        assert result["pickup_day_of_week"] == 4  # Friday
    
    def test_real_example_2(self):
        """Test with real example including categories."""
        html = """
        <div class="bold">
        <div class="bold">
        <div class="bold"><span style="font-size: 12pt;"><strong>Pickup: Saturday, March 14 <span title="Eastern Standard Time, UTC -5 hours">EDT</span>,  9AM - 12 NOON</strong></span></div>
        <div class="bold"><span style="font-size: 12pt;"><strong>Category A: 9AM - 11AM</strong></span></div>
        <div class="bold"><span style="font-size: 12pt;"><strong>Category B: 11AM - 12NOON</strong></span></div>
        <div class="bold"><span style="font-size: 12pt;"><strong>Pickup Location: 133 Beresford Ave, Toronto, ON M6S 3B2 <a href="https://www.google.com/maps/place/133+Beresford+Ave,+Toronto,+ON+M6S+3B2,+Canada/@43.6496864,-79.4766789,17z/data=!3m1!4b1!4m2!3m1!1s0x882b3687378b1f05:0x7498f5bd63afb521" target="_blank" rel="nofollow">Map</a></strong></span></div>
        </div>
        </div>
        """
        
        result = extract_pickup_windows(html)
        
        assert result["has_category_windows"] is True
        assert result["pickup_day_of_week"] == 5  # Saturday
        assert result["num_pickup_windows"] >= 2
    
    def test_am_to_pm_window(self):
        """Test window spanning AM to PM."""
        html = """
        <strong>Pickup: Sunday, March 29 EDT, 10AM - 2PM</strong>
        """
        
        result = extract_pickup_windows(html)
        
        assert result["total_pickup_hours"] == 4.0
        assert result["first_pickup_start_hour"] == 10.0
        assert result["last_pickup_end_hour"] == 14.0


class TestEngineerAuctionFeatures:
    """Test the engineer_auction_features function with pickup windows."""
    
    def test_engineer_auction_features_with_pickup_info(self):
        """Test that auction features include pickup window features."""
        df = pd.DataFrame({
            "auction_id": [1, 2, 3],
            "auction_removal_info": [
                "<strong>Pickup: Friday, March 13 EDT, 4PM - 7PM</strong>",
                "<strong>Pickup: Saturday, March 14 EDT, 9AM - 12 NOON</strong>",
                None,
            ]
        })
        
        result = engineer_auction_features(df)
        
        # Check that pickup window features are added
        assert "num_pickup_windows" in result.columns
        assert "total_pickup_hours" in result.columns
        assert "first_pickup_start_hour" in result.columns
        assert "last_pickup_end_hour" in result.columns
        assert "pickup_day_of_week" in result.columns
        assert "has_category_windows" in result.columns
        
        # Check first row
        assert result.loc[0, "num_pickup_windows"] == 1
        assert result.loc[0, "total_pickup_hours"] == 3.0
        assert result.loc[0, "pickup_day_of_week"] == 4  # Friday
        
        # Check second row
        assert result.loc[1, "num_pickup_windows"] == 1
        assert result.loc[1, "total_pickup_hours"] == 3.0
        assert result.loc[1, "pickup_day_of_week"] == 5  # Saturday
        
        # Check third row (None input)
        assert result.loc[2, "num_pickup_windows"] == 0
        assert result.loc[2, "total_pickup_hours"] == 0.0
        assert pd.isna(result.loc[2, "pickup_day_of_week"])
    
    def test_engineer_auction_features_without_pickup_column(self):
        """Test that function works when auction_removal_info column is missing."""
        df = pd.DataFrame({
            "auction_id": [1, 2, 3],
            "some_other_column": ["a", "b", "c"]
        })
        
        # Should not crash when column is missing
        result = engineer_auction_features(df)
        
        # Original columns should be preserved
        assert "auction_id" in result.columns
        assert "some_other_column" in result.columns
        
        # Pickup features should not be added
        assert "num_pickup_windows" not in result.columns
