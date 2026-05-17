import numpy as np

def predict_storage_overflow(current_count, capacity, avg_items_per_day, simulations=10000):
    days_until_full = []

    for _ in range(simulations):
        count = current_count
        days = 0
        while count < capacity:
            daily_intake = np.random.poisson(avg_items_per_day)
            count += daily_intake
            days += 1
            if days > 365:
                break
        days_until_full.append(days)

    avg_days = np.mean(days_until_full)
    min_days = int(np.min(days_until_full))
    max_days = int(np.max(days_until_full))

    return {
        'avg_days_until_full': round(avg_days, 1),
        'min_days': min_days,
        'max_days': max_days,
        'risk_level': 'HIGH' if avg_days <= 7 else 'MEDIUM' if avg_days <= 30 else 'LOW'
    }