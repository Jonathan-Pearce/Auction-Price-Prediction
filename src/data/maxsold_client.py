# =============================================================================
# Auction Price Prediction - MaxSold API Client
# =============================================================================
"""
API client for MaxSold auction data.

Endpoints:
1. Auction items list: /msapi/auctions/items?auctionid={id}&limit={n}
2. Item + bid history: /msapi/auctions/items?auctionid={id}&itemid={item_id}
3. Enriched item info: /listings/am/{item_id}/enriched

Features:
- Async HTTP client (httpx)
- Rate limiting
- Retry logic with exponential backoff
- Response validation with Pydantic
"""

import asyncio
from typing import Any

import httpx
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import settings
from src.data.schemas import (
    Auction,
    Item,
    Bid,
    AuctionResponse,
    ItemDetailResponse,
    EnrichedItemResponse,
)


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_second: int = 10):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time

            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)

            self.last_request_time = asyncio.get_event_loop().time()


# =============================================================================
# MaxSold API Client
# =============================================================================


class MaxSoldClient:
    """
    Async client for MaxSold API.

    Usage:
        async with MaxSoldClient() as client:
            items = await client.get_auction_items(auction_id=99941)
            item_detail = await client.get_item_detail(auction_id=99941, item_id=123)
    """

    def __init__(
        self,
        rate_limit: int | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = settings.maxsold.base_url
        self.enriched_base_url = settings.maxsold.enriched_base_url
        self.timeout = timeout

        # Rate limiting
        rate = rate_limit or settings.maxsold.rate_limit
        self.rate_limiter = RateLimiter(rate)

        # Retry settings
        self.retry_attempts = settings.maxsold.retry_attempts

        # HTTP client (initialized in __aenter__)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "MaxSoldClient":
        """Initialize async HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            headers={
                "User-Agent": "AuctionPricePredictor/1.0 (Research Project)",
                "Accept": "application/json",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        """Get HTTP client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with MaxSoldClient():'")
        return self._client

    # =========================================================================
    # API Methods
    # =========================================================================

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    )
    async def _request(
        self,
        url: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make rate-limited HTTP request with retry logic.

        Args:
            url: Full URL to request
            params: Query parameters

        Returns:
            JSON response as dictionary
        """
        await self.rate_limiter.acquire()

        logger.debug(f"Requesting: {url} with params: {params}")
        response = await self.client.get(url, params=params)
        response.raise_for_status()

        return response.json()

    async def get_auction_items(
        self,
        auction_id: int,
        limit: int | None = None,
    ) -> list[Item]:
        """
        Get all items for an auction.

        Args:
            auction_id: MaxSold auction ID
            limit: Maximum items to retrieve (default: 2500)

        Returns:
            List of Item objects
        """
        limit = limit or settings.maxsold.items_limit
        url = f"{self.base_url}/auctions/items"
        params = {"auctionid": auction_id, "limit": limit}

        try:
            data = await self._request(url, params)

            # Parse response - structure may vary, adapt as needed
            items = []
            if isinstance(data, list):
                items = [Item(**item_data) for item_data in data]
            elif isinstance(data, dict) and "items" in data:
                items = [Item(**item_data) for item_data in data["items"]]
            else:
                logger.warning(f"Unexpected response structure for auction {auction_id}")

            logger.info(f"Retrieved {len(items)} items for auction {auction_id}")
            return items

        except Exception as e:
            logger.error(f"Failed to get items for auction {auction_id}: {e}")
            raise

    async def get_item_detail(
        self,
        auction_id: int,
        item_id: int,
    ) -> ItemDetailResponse:
        """
        Get item details including bid history.

        Args:
            auction_id: MaxSold auction ID
            item_id: MaxSold item ID

        Returns:
            ItemDetailResponse with item and bids
        """
        url = f"{self.base_url}/auctions/items"
        params = {"auctionid": auction_id, "itemid": item_id}

        try:
            data = await self._request(url, params)

            # Parse item and bid history
            # Structure may vary - adapt based on actual API response
            item_data = data if isinstance(data, dict) else data[0] if data else {}

            item = Item(**item_data)
            bids = []

            # Extract bid history if present
            if "bids" in item_data:
                bids = [Bid(**bid_data) for bid_data in item_data["bids"]]
            elif "bidHistory" in item_data:
                bids = [Bid(**bid_data) for bid_data in item_data["bidHistory"]]

            logger.debug(f"Retrieved item {item_id} with {len(bids)} bids")
            return ItemDetailResponse(item=item, bids=bids)

        except Exception as e:
            logger.error(f"Failed to get detail for item {item_id}: {e}")
            raise

    async def get_enriched_item(
        self,
        item_id: int,
    ) -> EnrichedItemResponse:
        """
        Get enriched item information.

        Args:
            item_id: MaxSold item ID

        Returns:
            EnrichedItemResponse with additional data
        """
        url = f"{self.enriched_base_url}/listings/am/{item_id}/enriched"

        try:
            data = await self._request(url)

            logger.debug(f"Retrieved enriched data for item {item_id}")
            return EnrichedItemResponse(item_id=item_id, enriched_data=data)

        except Exception as e:
            logger.error(f"Failed to get enriched data for item {item_id}: {e}")
            raise

    # =========================================================================
    # Batch Methods
    # =========================================================================

    async def get_auction_with_bids(
        self,
        auction_id: int,
        include_enriched: bool = False,
    ) -> Auction:
        """
        Get full auction data including bid history for all items.

        Args:
            auction_id: MaxSold auction ID
            include_enriched: Whether to fetch enriched data for each item

        Returns:
            Auction object with all items and their bids
        """
        # Get all items
        items = await self.get_auction_items(auction_id)

        # Get bid history for each item
        items_with_bids = []
        for item in items:
            try:
                detail = await self.get_item_detail(auction_id, item.item_id)
                item.bids = detail.bids

                if include_enriched:
                    enriched = await self.get_enriched_item(item.item_id)
                    item.enriched_data = enriched.enriched_data

                items_with_bids.append(item)

            except Exception as e:
                logger.warning(f"Failed to get bids for item {item.item_id}: {e}")
                items_with_bids.append(item)  # Include item without bids

        auction = Auction(
            auction_id=auction_id,
            title=f"Auction {auction_id}",  # Placeholder - may need separate API call
            items=items_with_bids,
            num_items=len(items_with_bids),
        )

        logger.info(
            f"Retrieved auction {auction_id} with {len(items_with_bids)} items"
        )
        return auction

    async def get_multiple_auctions(
        self,
        auction_ids: list[int],
        include_bids: bool = True,
        include_enriched: bool = False,
        max_concurrent: int = 5,
    ) -> list[Auction]:
        """
        Get data for multiple auctions with controlled concurrency.

        Args:
            auction_ids: List of auction IDs to fetch
            include_bids: Whether to fetch bid history
            include_enriched: Whether to fetch enriched data
            max_concurrent: Maximum concurrent auction fetches

        Returns:
            List of Auction objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_auction(auction_id: int) -> Auction | None:
            async with semaphore:
                try:
                    if include_bids:
                        return await self.get_auction_with_bids(
                            auction_id, include_enriched
                        )
                    else:
                        items = await self.get_auction_items(auction_id)
                        return Auction(
                            auction_id=auction_id,
                            title=f"Auction {auction_id}",
                            items=items,
                            num_items=len(items),
                        )
                except Exception as e:
                    logger.error(f"Failed to fetch auction {auction_id}: {e}")
                    return None

        tasks = [fetch_auction(aid) for aid in auction_ids]
        results = await asyncio.gather(*tasks)

        # Filter out failed fetches
        auctions = [a for a in results if a is not None]
        logger.info(f"Successfully fetched {len(auctions)}/{len(auction_ids)} auctions")

        return auctions


# =============================================================================
# Sync Wrapper (for non-async contexts)
# =============================================================================


def get_auction_items_sync(auction_id: int) -> list[Item]:
    """Synchronous wrapper for get_auction_items."""

    async def _fetch():
        async with MaxSoldClient() as client:
            return await client.get_auction_items(auction_id)

    return asyncio.run(_fetch())


def get_item_detail_sync(auction_id: int, item_id: int) -> ItemDetailResponse:
    """Synchronous wrapper for get_item_detail."""

    async def _fetch():
        async with MaxSoldClient() as client:
            return await client.get_item_detail(auction_id, item_id)

    return asyncio.run(_fetch())
