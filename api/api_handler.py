"""
Dynamic API handler for OutScan interactive demo
Returns live-style data with randomization to simulate real system
"""
import json
import random
from datetime import datetime, timezone, timedelta

def lambda_handler(event, context):
    """
    Handle API requests for OutScan demo with dynamic data
    """
    
    # Dynamic base values with realistic variation
    base_sequences = 125000
    base_variants = 15
    
    # Randomize current values to simulate live updates
    current_sequences = base_sequences + random.randint(-500, 500)
    current_variants = base_variants + random.choice([-1, -1, 0, 0, 1])
    processing_rate = random.randint(2800, 3200)
    active_alerts = random.choice([2, 3, 4, 5])
    
    # Generate dynamic daily processing data for last 7 days
    daily_processing = []
    for i in range(7):
        date = (datetime.now(timezone.utc) - timedelta(days=6-i))
        daily_processing.append({
            "date": date.strftime("%b %d"),
            "count": random.randint(17500, 22000)
        })
    
    # Generate dynamic variant prevalence data
    variant_prevalence = [
        {"name": "XBB.1.5", "percentage": random.randint(30, 38)},
        {"name": "BQ.1.1", "percentage": random.randint(22, 28)},
        {"name": "XBB.2.3", "percentage": random.randint(15, 20)},
        {"name": "BA.5", "percentage": random.randint(8, 15)},
        {"name": "Others", "percentage": random.randint(5, 10)}
    ]
    
    # Ensure percentages add up to 100
    total = sum(v["percentage"] for v in variant_prevalence)
    if total != 100:
        variant_prevalence[0]["percentage"] += (100 - total)
    
    # Response data matching dashboard expectations
    response_data = {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sequences_analyzed": current_sequences,
        "variants_detected": current_variants,
        "active_alerts": active_alerts,
        "processing_rate_per_minute": processing_rate,
        "daily_processing": daily_processing,
        "variant_prevalence": variant_prevalence,
        "system_info": {
            "uptime": f"{random.uniform(99.7, 99.9):.1f}%",
            "avg_response_time": f"{random.randint(98, 145)}ms",
            "total_countries": random.randint(45, 50),
            "accuracy_rate": f"{random.uniform(94.2, 95.8):.1f}%"
        }
    }
    
    # Return with proper CORS headers for browser access
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
            'Access-Control-Allow-Methods': 'GET,OPTIONS,POST,PUT,DELETE'
        },
        'body': json.dumps(response_data, indent=2)
    }

# Local testing function
if __name__ == "__main__":
    test_event = {}
    test_context = type('Context', (), {'aws_request_id': 'test-request-id'})()
    
    result = lambda_handler(test_event, test_context)
    print("API Response:")
    print(result['body']) 