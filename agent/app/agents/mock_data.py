"""Mock data source for travel assistant."""

MOCK_FLIGHTS = [
    {"route": "hcm-hanoi", "airline": "VietJet", "price_vnd": 1_250_000, "duration": "2h10"},
    {"route": "hcm-hanoi", "airline": "Vietnam Airlines", "price_vnd": 1_950_000, "duration": "2h05"},
    {"route": "hcm-danang", "airline": "Bamboo", "price_vnd": 1_050_000, "duration": "1h25"},
]

MOCK_HOTELS = [
    {"city": "hanoi", "name": "Hanoi Cozy Stay", "price_vnd_per_night": 650_000, "rating": 4.2},
    {"city": "hanoi", "name": "Lake View Hotel", "price_vnd_per_night": 980_000, "rating": 4.5},
    {"city": "danang", "name": "Danang Beach Inn", "price_vnd_per_night": 720_000, "rating": 4.3},
]

MOCK_ITINERARY = {
    "hanoi": [
        "Day 1: Hoan Kiem Lake, Old Quarter food tour, Train Street coffee",
        "Day 2: Temple of Literature, Ho Chi Minh Mausoleum, West Lake sunset",
        "Day 3: Bat Trang village or Ninh Binh day trip",
    ],
    "danang": [
        "Day 1: My Khe beach, Han market, Dragon Bridge night show",
        "Day 2: Ba Na Hills, Golden Bridge",
        "Day 3: Son Tra peninsula, seafood by the sea",
    ],
}
