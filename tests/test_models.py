from datetime import datetime
from app.bayesian import bayesian_match_score

found_item = {
    'category': 'bag',
    'zone_id': 2,
    'found_at': datetime(2024, 1, 15, 10, 0)
}

lost_report = {
    'category': 'bag',
    'zone_id': 3,
    'lost_at': datetime(2024, 1, 15, 8, 0)
}

score = bayesian_match_score(found_item, lost_report)
print(f"Match probability: {score}%")

from app.montecarlo import predict_storage_overflow

result = predict_storage_overflow(
    current_count=40,
    capacity=50,
    avg_items_per_day=3
)
print(f"Storage prediction: {result}")