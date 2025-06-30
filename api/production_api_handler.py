"""
Production API Handler for OutScan Genomic Early Warning System
Provides secure, validated, and monitored API endpoints for external integrations
"""
import json
import boto3
import logging
import os
import hashlib
import hmac
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DecimalEncoder(json.JSONEncoder):
    """JSON encoder for DynamoDB Decimal types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

class ProductionAPIHandler:
    """
    Production-grade API handler with security, validation, and monitoring
    """
    
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.cloudwatch = boto3.client('cloudwatch')
        
        # Environment configuration
        self.variant_table_name = os.environ.get('VARIANT_TABLE_NAME', 'OutScan-VariantTable')
        self.sequence_table_name = os.environ.get('SEQUENCE_TABLE_NAME', 'OutScan-SequenceTable')
        self.alert_table_name = os.environ.get('ALERT_TABLE_NAME', 'OutScan-AlertTable')
        self.api_key_hash = os.environ.get('API_KEY_HASH', '')
        self.rate_limit_per_minute = int(os.environ.get('RATE_LIMIT_PER_MINUTE', '100'))
        
        # Initialize tables
        self.variant_table = self.dynamodb.Table(self.variant_table_name)
        self.sequence_table = self.dynamodb.Table(self.sequence_table_name)
        self.alert_table = self.dynamodb.Table(self.alert_table_name)
        
        # In-memory rate limiting (would use Redis/ElastiCache in real production)
        self.request_counts = {}
    
    def authenticate_request(self, event: Dict) -> bool:
        """
        Validate API key authentication
        """
        headers = event.get('headers', {})
        api_key = headers.get('X-API-Key') or headers.get('x-api-key')
        
        if not api_key:
            logger.warning("Missing API key in request")
            return False
        
        # Hash the provided key and compare with stored hash
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        if key_hash != self.api_key_hash:
            logger.warning(f"Invalid API key provided: {key_hash[:8]}...")
            return False
        
        return True
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """
        Basic rate limiting (in production, use Redis/ElastiCache)
        """
        current_minute = int(time.time() // 60)
        
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {}
        
        # Clean old entries
        old_minutes = [minute for minute in self.request_counts[client_ip] 
                      if minute < current_minute - 1]
        for minute in old_minutes:
            del self.request_counts[client_ip][minute]
        
        # Check current minute count
        current_count = self.request_counts[client_ip].get(current_minute, 0)
        
        if current_count >= self.rate_limit_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False
        
        # Increment count
        self.request_counts[client_ip][current_minute] = current_count + 1
        return True
    
    def validate_variant_query(self, query_params: Dict) -> Dict:
        """
        Validate query parameters for variant endpoint
        """
        errors = []
        validated_params = {}
        
        # Validate limit parameter
        limit = query_params.get('limit', ['50'])[0]
        try:
            limit = int(limit)
            if limit < 1 or limit > 1000:
                errors.append("Limit must be between 1 and 1000")
            else:
                validated_params['limit'] = limit
        except ValueError:
            errors.append("Limit must be a valid integer")
        
        # Validate date range
        start_date = query_params.get('start_date', [None])[0]
        end_date = query_params.get('end_date', [None])[0]
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                validated_params['start_date'] = start_dt
            except ValueError:
                errors.append("start_date must be in ISO format (YYYY-MM-DDTHH:MM:SSZ)")
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                validated_params['end_date'] = end_dt
            except ValueError:
                errors.append("end_date must be in ISO format (YYYY-MM-DDTHH:MM:SSZ)")
        
        # Validate risk level filter
        risk_level = query_params.get('risk_level', [None])[0]
        if risk_level and risk_level not in ['LOW', 'MEDIUM', 'HIGH']:
            errors.append("risk_level must be one of: LOW, MEDIUM, HIGH")
        else:
            validated_params['risk_level'] = risk_level
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'params': validated_params
        }
    
    def get_variants(self, validated_params: Dict) -> Dict:
        """
        Query variants from DynamoDB with filtering
        """
        try:
            # Build scan parameters
            scan_params = {
                'Limit': validated_params.get('limit', 50)
            }
            
            # Add filter expressions if needed
            filter_expressions = []
            expression_values = {}
            
            if 'start_date' in validated_params:
                filter_expressions.append('#ts >= :start_date')
                expression_values[':start_date'] = validated_params['start_date'].timestamp()
            
            if 'end_date' in validated_params:
                filter_expressions.append('#ts <= :end_date')
                expression_values[':end_date'] = validated_params['end_date'].timestamp()
            
            if 'risk_level' in validated_params and validated_params['risk_level']:
                filter_expressions.append('#risk = :risk_level')
                expression_values[':risk_level'] = validated_params['risk_level']
            
            if filter_expressions:
                scan_params['FilterExpression'] = ' AND '.join(filter_expressions)
                scan_params['ExpressionAttributeValues'] = expression_values
                scan_params['ExpressionAttributeNames'] = {'#ts': 'timestamp', '#risk': 'risk_level'}
            
            # Execute query
            response = self.variant_table.scan(**scan_params)
            
            # Process results
            variants = []
            for item in response.get('Items', []):
                # Convert DynamoDB item to API response format
                variant = {
                    'variant_id': item.get('variant_id'),
                    'mutations': item.get('mutations', []),
                    'countries_detected': item.get('countries', []),
                    'detection_date': item.get('detection_date'),
                    'risk_score': float(item.get('risk_score', 0)),
                    'growth_rate': float(item.get('growth_rate', 0)),
                    'cluster_size': int(item.get('cluster_size', 1))
                }
                variants.append(variant)
            
            return {
                'success': True,
                'data': {
                    'variants': variants,
                    'total_count': len(variants),
                    'has_more': 'LastEvaluatedKey' in response
                }
            }
            
        except Exception as e:
            logger.error(f"Error querying variants: {str(e)}")
            return {
                'success': False,
                'error': 'Database query failed',
                'details': str(e)
            }
    
    def get_system_status(self) -> Dict:
        """
        Get current system status and statistics
        """
        try:
            # Get recent statistics
            current_time = datetime.now(timezone.utc)
            yesterday = current_time - timedelta(days=1)
            
            # Query recent sequences
            sequence_response = self.sequence_table.scan(
                FilterExpression='#ts >= :yesterday',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':yesterday': yesterday.timestamp()},
                Select='COUNT'
            )
            
            # Query active alerts
            alert_response = self.alert_table.scan(
                FilterExpression='#status = :active',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':active': 'ACTIVE'},
                Select='COUNT'
            )
            
            # Calculate statistics
            sequences_24h = sequence_response.get('Count', 0)
            active_alerts = alert_response.get('Count', 0)
            
            return {
                'success': True,
                'data': {
                    'status': 'operational',
                    'timestamp': current_time.isoformat(),
                    'sequences_processed_24h': sequences_24h,
                    'active_alerts': active_alerts,
                    'system_version': '1.0.0',
                    'uptime_percentage': 99.8  # Would come from CloudWatch in production
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'success': False,
                'error': 'Failed to retrieve system status',
                'details': str(e)
            }
    
    def log_api_metrics(self, endpoint: str, method: str, status_code: int, 
                       response_time_ms: int, client_ip: str):
        """
        Log metrics to CloudWatch
        """
        try:
            self.cloudwatch.put_metric_data(
                Namespace='OutScan/API',
                MetricData=[
                    {
                        'MetricName': 'RequestCount',
                        'Dimensions': [
                            {'Name': 'Endpoint', 'Value': endpoint},
                            {'Name': 'Method', 'Value': method},
                            {'Name': 'StatusCode', 'Value': str(status_code)}
                        ],
                        'Value': 1,
                        'Unit': 'Count'
                    },
                    {
                        'MetricName': 'ResponseTime',
                        'Dimensions': [
                            {'Name': 'Endpoint', 'Value': endpoint}
                        ],
                        'Value': response_time_ms,
                        'Unit': 'Milliseconds'
                    }
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to log metrics: {str(e)}")

def lambda_handler(event, context):
    """
    Main Lambda handler for production API
    """
    start_time = time.time()
    handler = ProductionAPIHandler()
    
    try:
        # Extract request details
        http_method = event.get('httpMethod', 'GET')
        path = event.get('path', '/')
        client_ip = event.get('requestContext', {}).get('identity', {}).get('sourceIp', 'unknown')
        
        logger.info(f"API Request: {http_method} {path} from {client_ip}")
        
        # Common security headers
        headers = {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-API-Key,Authorization',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
        }
        
        # Handle OPTIONS (CORS preflight)
        if http_method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': headers,
                'body': ''
            }
        
        # Rate limiting
        if not handler.check_rate_limit(client_ip):
            return {
                'statusCode': 429,
                'headers': headers,
                'body': json.dumps({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {handler.rate_limit_per_minute} requests per minute'
                })
            }
        
        # Authentication for protected endpoints
        if path.startswith('/protected/'):
            if not handler.authenticate_request(event):
                return {
                    'statusCode': 401,
                    'headers': headers,
                    'body': json.dumps({
                        'error': 'Unauthorized',
                        'message': 'Valid API key required'
                    })
                }
        
        # Route handling
        if path == '/variants' and http_method == 'GET':
            # Validate query parameters
            query_params = event.get('queryStringParameters') or {}
            validation = handler.validate_variant_query(query_params)
            
            if not validation['valid']:
                return {
                    'statusCode': 400,
                    'headers': headers,
                    'body': json.dumps({
                        'error': 'Invalid parameters',
                        'details': validation['errors']
                    })
                }
            
            # Get variants
            result = handler.get_variants(validation['params'])
            status_code = 200 if result['success'] else 500
            
        elif path == '/status' and http_method == 'GET':
            # Get system status
            result = handler.get_system_status()
            status_code = 200 if result['success'] else 500
            
        else:
            # Unknown endpoint
            result = {
                'success': False,
                'error': 'Not Found',
                'message': f'Endpoint {http_method} {path} not found'
            }
            status_code = 404
        
        # Log metrics
        response_time_ms = int((time.time() - start_time) * 1000)
        handler.log_api_metrics(path, http_method, status_code, response_time_ms, client_ip)
        
        # Prepare response
        response_body = result
        if not result.get('success', True):
            # Remove internal error details from client response
            if 'details' in response_body and status_code >= 500:
                response_body = {
                    'error': result.get('error', 'Internal server error'),
                    'message': 'An internal error occurred'
                }
        
        return {
            'statusCode': status_code,
            'headers': headers,
            'body': json.dumps(response_body, cls=DecimalEncoder, indent=2)
        }
        
    except Exception as e:
        logger.error(f"Unhandled error in API handler: {str(e)}")
        
        # Log error metrics
        response_time_ms = int((time.time() - start_time) * 1000)
        handler.log_api_metrics(
            event.get('path', '/unknown'), 
            event.get('httpMethod', 'UNKNOWN'), 
            500, response_time_ms, 
            event.get('requestContext', {}).get('identity', {}).get('sourceIp', 'unknown')
        )
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred'
            })
        }

# Local testing
if __name__ == "__main__":
    test_event = {
        'httpMethod': 'GET',
        'path': '/status',
        'queryStringParameters': None,
        'headers': {},
        'requestContext': {
            'identity': {'sourceIp': '127.0.0.1'}
        }
    }
    
    class Context:
        aws_request_id = 'test-request'
    
    result = lambda_handler(test_event, Context())
    print(json.dumps(result, indent=2)) 