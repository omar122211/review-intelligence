"""Labels used everywhere: overall sentiment and the five aspects campers care about."""

SENTIMENTS = ["negative", "neutral", "positive"]

# campground aspect → column with the reviewer's own 1-5 star rating in the TripAdvisor data
ASPECTS = {
    "cleanliness": "cleanliness",
    "value": "value",
    "location": "location",
    "facilities": "rooms",          # hotel "rooms" ≈ campsites, bathrooms, showers
    "noise_sleep": "sleep_quality",  # "sleep quality" ≈ quiet nights
}
ASPECT_LABELS = {
    "cleanliness": "Cleanliness",
    "value": "Value for money",
    "location": "Location",
    "facilities": "Sites & facilities",
    "noise_sleep": "Quiet & sleep",
}


def stars_to_sentiment(stars: float) -> str:
    """1-2 stars → negative, 3 → neutral, 4-5 → positive."""
    return "negative" if stars <= 2 else "neutral" if stars < 4 else "positive"
