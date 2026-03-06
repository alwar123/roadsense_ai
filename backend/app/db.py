import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "roadsense")
MONGO_HAZARDS_COLLECTION = os.getenv("MONGO_HAZARDS_COLLECTION", "hazards")


_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """
    Return a singleton Motor client instance.
    """
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client


def get_hazards_collection() -> AsyncIOMotorCollection:
    client = get_client()
    db = client[MONGO_DB_NAME]
    return db[MONGO_HAZARDS_COLLECTION]


async def insert_hazard(
    hazard_type: str,
    latitude: float,
    longitude: float,
    timestamp: datetime | None = None,
) -> None:
    """
    Insert a single hazard document into the hazards collection.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    collection = get_hazards_collection()
    doc: Dict[str, Any] = {
        "hazard_type": hazard_type,
        "latitude": latitude,
        "longitude": longitude,
        "timestamp": timestamp,
    }
    await collection.insert_one(doc)


async def fetch_hazards() -> List[Dict[str, Any]]:
    """
    Fetch all hazards, converting ObjectId and datetime to JSON-friendly values.
    """
    collection = get_hazards_collection()
    cursor = collection.find().sort("timestamp", 1)
    results: List[Dict[str, Any]] = []

    async for doc in cursor:
        doc_id = str(doc.get("_id"))
        ts = doc.get("timestamp")
        if isinstance(ts, datetime):
            ts_iso = ts.astimezone(timezone.utc).isoformat()
        else:
            ts_iso = str(ts)

        results.append(
            {
                "id": doc_id,
                "hazard_type": doc.get("hazard_type"),
                "latitude": doc.get("latitude"),
                "longitude": doc.get("longitude"),
                "timestamp": ts_iso,
            }
        )

    return results

