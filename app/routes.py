from flask import Blueprint, request, jsonify
from datetime import datetime
from app.bayesian import bayesian_match_score
from app.montecarlo import predict_storage_overflow

bp = Blueprint('routes', __name__)


@bp.route('/match', methods=['POST'])
def match():
    try:
        data = request.get_json()
        found_item = {
            'category_id': data['found_item']['category_id'],
            'zone_id':     data['found_item']['zone_id'],
            'found_at':    datetime.fromisoformat(data['found_item']['found_at']),
        }
        lost_report = {
            'category_id': data['lost_report']['category_id'],
            'zone_id':     data['lost_report']['zone_id'],
            'lost_at':     datetime.fromisoformat(data['lost_report']['lost_at']),
        }
        score = bayesian_match_score(found_item, lost_report)
        return jsonify({
            'success':     True,
            'probability': score,
            'label':       'HIGH' if score >= 70 else 'MEDIUM' if score >= 40 else 'LOW',
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@bp.route('/predict-storage', methods=['POST'])
def predict_storage():
    try:
        data = request.get_json()
        result = predict_storage_overflow(
            current_count=data['current_count'],
            capacity=data['capacity'],
            avg_items_per_day=data['avg_items_per_day']
        )
        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@bp.route('/identify', methods=['POST'])
def identify():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        image_bytes = request.files['image'].read()
        from app.vision import identify_item
        result = identify_item(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@bp.route('/found-report', methods=['POST'])
def create_found_report():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['title', 'description', 'category_id', 'date_occurred', 'location']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        found_report = {
            'title': data['title'],
            'description': data['description'],
            'category_id': data['category_id'],
            'date_occurred': datetime.fromisoformat(data['date_occurred']),
            'location': data['location'],
            'posting_type': data.get('posting_type', 'hold'),
        }
        
        # Find matching lost reports
        ml_matches = []
        if data.get('match_with_lost', True):
            # This would need lost reports from your database
            # For now, returning empty matches structure
            pass
        
        return jsonify({
            'success': True,
            'message': 'Found report created successfully',
            'data': {
                'report': found_report,
                'matches': ml_matches,
            }
        }), 201
        
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
        