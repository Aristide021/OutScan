"""
SNS Alert Dispatcher
Multi-channel notification system for variant alerts
"""
import json
import boto3
import logging
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AlertLevel(Enum):
    RED = "RED_ALERT"
    AMBER = "AMBER_ALERT" 
    MONITORING = "MONITORING"

class SNSAlertDispatcher:
    """
    Dispatches alerts via multiple channels using Amazon SNS
    """
    
    def __init__(self):
        self.sns_client = boto3.client('sns')
        self.dynamodb = boto3.resource('dynamodb')
        self.alert_table = self.dynamodb.Table('AlertHistory')
        
        # SNS Topic ARNs (would be configured via environment)
        self.topics = {
            'WHO_ALERTS': 'arn:aws:sns:us-east-1:123456789012:who-alerts',
            'HEALTH_AUTHORITIES': 'arn:aws:sns:us-east-1:123456789012:health-authorities',
            'RESEARCH_COMMUNITY': 'arn:aws:sns:us-east-1:123456789012:research-alerts',
            'PUBLIC_DASHBOARD': 'arn:aws:sns:us-east-1:123456789012:public-alerts'
        }
    
    def dispatch_variant_alert(self, alert_data: Dict) -> Dict:
        """
        Dispatch variant alert to appropriate channels
        """
        try:
            alert_level = AlertLevel(alert_data.get('alert_level', 'MONITORING'))
            variant_info = alert_data.get('variant_info', {})
            risk_assessment = alert_data.get('risk_assessment', {})
            
            # Generate alert messages
            messages = self._generate_alert_messages(alert_level, variant_info, risk_assessment)
            
            # Dispatch to channels based on alert level
            dispatch_results = {}
            
            if alert_level == AlertLevel.RED:
                # RED alerts go to all channels
                for channel, topic_arn in self.topics.items():
                    result = self._send_sns_alert(
                        topic_arn, 
                        messages[channel], 
                        alert_level.value,
                        variant_info.get('variant_id', 'Unknown')
                    )
                    dispatch_results[channel] = result
                    
            elif alert_level == AlertLevel.AMBER:
                # AMBER alerts to health authorities and research
                priority_channels = ['WHO_ALERTS', 'HEALTH_AUTHORITIES', 'RESEARCH_COMMUNITY']
                for channel in priority_channels:
                    result = self._send_sns_alert(
                        self.topics[channel],
                        messages[channel],
                        alert_level.value,
                        variant_info.get('variant_id', 'Unknown')
                    )
                    dispatch_results[channel] = result
            
            else:
                # MONITORING alerts to research community only
                result = self._send_sns_alert(
                    self.topics['RESEARCH_COMMUNITY'],
                    messages['RESEARCH_COMMUNITY'],
                    alert_level.value,
                    variant_info.get('variant_id', 'Unknown')
                )
                dispatch_results['RESEARCH_COMMUNITY'] = result
            
            # Store dispatch record
            self._store_dispatch_record(alert_data, dispatch_results)
            
            logger.info(f"Alert dispatched: {alert_level.value} - {len(dispatch_results)} channels")
            
            return {
                'status': 'SUCCESS',
                'alert_level': alert_level.value,
                'channels_notified': list(dispatch_results.keys()),
                'dispatch_results': dispatch_results
            }
            
        except Exception as e:
            logger.error(f"Error dispatching alert: {str(e)}")
            return {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def _generate_alert_messages(self, alert_level: AlertLevel, 
                               variant_info: Dict, risk_assessment: Dict) -> Dict:
        """
        Generate channel-specific alert messages
        """
        # Extract key information
        variant_id = variant_info.get('variant_id', 'Unknown')
        mutations = variant_info.get('key_mutations', [])
        countries = variant_info.get('countries_detected', [])
        growth_rate = variant_info.get('growth_rate', 0.0)
        
        risk_score = risk_assessment.get('composite_risk_score', 0.5)
        immune_escape = risk_assessment.get('mutation_analysis', {}).get('immune_escape', 0.0)
        
        # Base message components
        mutation_str = ', '.join(mutations[:5])  # Top 5 mutations
        country_str = ', '.join(countries[:5])   # Top 5 countries
        
        messages = {}
        
        # WHO Alert Message
        messages['WHO_ALERTS'] = f"""
[{alert_level.value}] New VOC Detected: {variant_id}

KEY DETAILS:
- Mutations: {mutation_str}
- Risk Score: {risk_score:.0%}
- Countries: {len(countries)} ({country_str})
- Growth Rate: {growth_rate:.1f}%/week
- Immune Escape: {immune_escape:.0%}

IMMEDIATE ACTIONS REQUIRED:
- Enhanced genomic surveillance
- Vaccine effectiveness monitoring
- International coordination

Report ID: WHO-{datetime.now().strftime('%Y%m%d')}-{variant_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
        """.strip()
        
        # Health Authorities Message
        messages['HEALTH_AUTHORITIES'] = f"""
🚨 VARIANT ALERT: {alert_level.value}

Variant: {variant_id}
Risk Level: {risk_score:.0%} 
Mutations: {mutation_str}
Geographic Spread: {len(countries)} countries
Weekly Growth: {growth_rate:.1f}%

RECOMMENDED ACTIONS:
{"- Immediate response protocols" if alert_level == AlertLevel.RED else "- Enhanced surveillance"}
- Laboratory preparedness
- Contact tracing protocols
{"- Airport screening activation" if alert_level == AlertLevel.RED else ""}

Contact: outscan-alerts@health.gov
Time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
        """.strip()
        
        # Research Community Message
        messages['RESEARCH_COMMUNITY'] = f"""
📊 VARIANT UPDATE: {variant_id}

Technical Details:
- Mutation Profile: {mutation_str}
- Phylogenetic Cluster: Growing {growth_rate:.1f}%/week
- Geographic Distribution: {country_str}
- Immune Escape Potential: {immune_escape:.0%}
- Alert Level: {alert_level.value}

Data Access:
- Sequences: GISAID EPI_ISL_XXXXXX
- Analysis: S3://outscan-public/variant-{variant_id}/
- Dashboard: https://outscan.health.gov/variant/{variant_id}

Generated: {datetime.now().isoformat()}
        """.strip()
        
        # Public Dashboard Message
        messages['PUBLIC_DASHBOARD'] = f"""
🔬 Variant Monitoring Update

A new variant ({variant_id}) is being monitored by health authorities.

Key Facts:
- Detection Date: {datetime.now().strftime('%B %d, %Y')}
- Countries Monitoring: {len(countries)}
- Status: Under Investigation

Health authorities are monitoring this variant closely. Continue following standard health guidelines.

More info: https://outscan.health.gov/public-dashboard
        """.strip()
        
        return messages
    
    def _send_sns_alert(self, topic_arn: str, message: str, 
                       alert_level: str, variant_id: str) -> Dict:
        """
        Send alert via SNS
        """
        try:
            # Message attributes for filtering
            message_attributes = {
                'AlertLevel': {
                    'DataType': 'String',
                    'StringValue': alert_level
                },
                'VariantID': {
                    'DataType': 'String', 
                    'StringValue': variant_id
                },
                'Timestamp': {
                    'DataType': 'String',
                    'StringValue': datetime.now().isoformat()
                }
            }
            
            # Send message
            response = self.sns_client.publish(
                TopicArn=topic_arn,
                Message=message,
                Subject=f"[{alert_level}] Variant Alert: {variant_id}",
                MessageAttributes=message_attributes
            )
            
            return {
                'status': 'SUCCESS',
                'message_id': response['MessageId'],
                'topic_arn': topic_arn
            }
            
        except Exception as e:
            logger.error(f"SNS publish failed for {topic_arn}: {str(e)}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'topic_arn': topic_arn
            }
    
    def _store_dispatch_record(self, alert_data: Dict, dispatch_results: Dict):
        """
        Store alert dispatch record in DynamoDB
        """
        try:
            record = {
                'AlertID': f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'DispatchTimestamp': datetime.now().isoformat(),
                'AlertLevel': alert_data.get('alert_level', 'MONITORING'),
                'VariantID': alert_data.get('variant_info', {}).get('variant_id', 'Unknown'),
                'ChannelsNotified': list(dispatch_results.keys()),
                'DispatchResults': json.dumps(dispatch_results),
                'AlertData': json.dumps(alert_data)
            }
            
            self.alert_table.put_item(Item=record)
            logger.info(f"Stored dispatch record: {record['AlertID']}")
            
        except Exception as e:
            logger.error(f"Error storing dispatch record: {str(e)}")

def lambda_handler(event, context):
    """
    AWS Lambda handler for alert dispatching
    """
    try:
        dispatcher = SNSAlertDispatcher()
        
        # Extract alert data from event
        alert_data = event.get('alert_data', {})
        
        if not alert_data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No alert data provided'})
            }
        
        # Dispatch alert
        result = dispatcher.dispatch_variant_alert(alert_data)
        
        return {
            'statusCode': 200 if result['status'] == 'SUCCESS' else 500,
            'body': json.dumps({
                'message': 'Alert dispatch completed',
                'result': result
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

if __name__ == "__main__":
    # For local testing
    test_event = {
        'alert_data': {
            'alert_level': 'RED_ALERT',
            'variant_info': {
                'variant_id': 'VAR-2025-001',
                'key_mutations': ['N501Y', 'E484K', 'K417N', 'L452R'],
                'countries_detected': ['United Kingdom', 'South Africa', 'United States'],
                'growth_rate': 35.2,
                'first_detection': '2025-06-15'
            },
            'risk_assessment': {
                'composite_risk_score': 0.84,
                'mutation_analysis': {
                    'immune_escape': 0.8,
                    'transmissibility': 1.4,
                    'virulence': 0.2
                }
            }
        }
    }
    
    test_context = type('Context', (), {'aws_request_id': 'test-request-id'})()
    
    result = lambda_handler(test_event, test_context)
    print(json.dumps(result, indent=2)) 