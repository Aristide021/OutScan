"""
Standalone API Handler for OutScan Variants Endpoint
Simple Lambda function to provide dynamic variant data
"""
import json
import random
from datetime import datetime, timedelta


def lambda_handler(event, context):
    """
    Handle GET /variants request with dynamic data
    """
    
    try:
        # Generate dynamic data based on current time
        now = datetime.utcnow()
        base_sequences = 120000 + random.randint(1000, 5000)
        
        # Generate recent daily processing data
        daily_data = []
        for i in range(7):
            date = now - timedelta(days=i)
            daily_data.append({
                "date": date.strftime("%b %d"),
                "count": random.randint(15000, 25000)
            })
        daily_data.reverse()
        
        # Generate variant data
        variants = [
            {
                "id": f"VOC-{i+1:03d}",
                "name": f"OutScan-{i+1}",
                "mutations": random.randint(8, 15),
                "sequences": random.randint(100, 1500),
                "countries": random.randint(5, 25),
                "growth_rate": round(random.uniform(0.8, 3.2), 1),
                "risk_level": random.choice(["LOW", "MEDIUM", "HIGH"])
            }
            for i in range(random.randint(12, 18))
        ]
        
        # Generate alert data
        alerts = [
            {
                "id": f"ALERT-{now.strftime('%Y%m%d')}-{i+1:02d}",
                "variant": f"VOC-{random.randint(1, 15):03d}",
                "level": random.choice(["AMBER", "RED", "MONITORING"]),
                "countries": random.randint(3, 12),
                "timestamp": (now - timedelta(hours=random.randint(1, 48))).isoformat()
            }
            for i in range(random.randint(2, 6))
        ]
        
        # Generate geographic distribution
        countries = [
            {"country": "United States", "sequences": random.randint(8000, 15000)},
            {"country": "United Kingdom", "sequences": random.randint(4000, 8000)},
            {"country": "Germany", "sequences": random.randint(3000, 6000)},
            {"country": "Canada", "sequences": random.randint(2000, 4000)},
            {"country": "France", "sequences": random.randint(2000, 4000)},
            {"country": "Australia", "sequences": random.randint(1500, 3000)},
            {"country": "Japan", "sequences": random.randint(1000, 2500)},
            {"country": "Brazil", "sequences": random.randint(1000, 2000)}
        ]
        
        # Main response data
        response_data = {
            "status": "operational",
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "sequences_analyzed": base_sequences + random.randint(-100, 500),
            "variants_detected": len(variants),
            "active_alerts": len([a for a in alerts if a["level"] in ["AMBER", "RED"]]),
            "processing_rate_per_minute": random.randint(2500, 3500),
            "daily_processing": daily_data,
            "variants": variants[:10],  # Top 10 variants
            "recent_alerts": alerts[:5],  # Most recent alerts
            "geographic_distribution": countries,
            "system_metrics": {
                "uptime_hours": random.randint(720, 8760),
                "data_freshness_minutes": random.randint(2, 15),
                "clustering_accuracy": round(random.uniform(94.5, 98.2), 1),
                "pipeline_success_rate": round(random.uniform(97.8, 99.9), 1)
            },
            "mutation_trends": [
                {
                    "position": pos,
                    "frequency": round(random.uniform(0.1, 15.8), 1),
                    "trend": random.choice(["increasing", "stable", "decreasing"])
                }
                for pos in ["L452R", "E484K", "N501Y", "D614G", "P681H", "K417N", "T478K"]
            ]
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization',
                'Cache-Control': 'no-cache'
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e),
                'status': 'error'
            })
        }


# Test function for local development
if __name__ == "__main__":
    test_event = {
        'httpMethod': 'GET',
        'path': '/variants'
    }
    
    result = lambda_handler(test_event, None)
    print(f"Status: {result['statusCode']}")
    print(f"Response: {result['body'][:200]}...") 