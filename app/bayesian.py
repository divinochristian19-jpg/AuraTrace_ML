ZONE_WEIGHT = 0.4
TIME_WEIGHT = 0.3
CATEGORY_WEIGHT = 0.3

def category_score(found_category_id, lost_category_id):
    return 1.0 if int(found_category_id) == int(lost_category_id) else 0.0

def zone_score(found_zone, reported_zone, max_distance=5):
    distance = abs(found_zone - reported_zone)
    if distance >= max_distance:
        return 0.0
    return 1.0 - (distance / max_distance)

def time_score(found_timestamp, reported_timestamp, max_hours=72):
    diff_hours = abs((found_timestamp - reported_timestamp).total_seconds()) / 3600
    if diff_hours >= max_hours:
        return 0.0
    return 1.0 - (diff_hours / max_hours)

def bayesian_match_score(found_item, lost_report):
    cat  = category_score(found_item['category_id'], lost_report['category_id'])
    zone = zone_score(found_item['zone_id'], lost_report['zone_id'])
    time = time_score(found_item['found_at'], lost_report['lost_at'])

    score = (CATEGORY_WEIGHT * cat) + (ZONE_WEIGHT * zone) + (TIME_WEIGHT * time)
    return round(score * 100, 2)