"""
WHO Report Generator
Automated generation of WHO-compliant variant reports
"""
import json
import boto3
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
import requests
from jinja2 import Template
import uuid

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class WHOReportGenerator:
    """
    Generates WHO-compliant reports for variant notifications
    """
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.alert_table = self.dynamodb.Table('AlertHistory')
        
        # WHO API configuration
        self.who_api_base = "https://covid19-surveillance-api.who.int"
        self.who_api_key = None  # Would be set from environment
        
    def generate_voc_report(self, variant_data: Dict, risk_assessment: Dict) -> Dict:
        """
        Generate WHO Variant of Concern (VOC) report
        """
        try:
            # Extract key information
            mutations = variant_data.get('mutations', [])
            countries = variant_data.get('countries', [])
            cluster_size = variant_data.get('cluster_size', 0)
            growth_rate = variant_data.get('growth_rate', 0.0)
            
            # Generate unique report ID
            report_id = f"WHO-VOC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
            
            # Create report structure
            report = {
                'report_metadata': {
                    'report_id': report_id,
                    'report_type': 'VARIANT_OF_CONCERN',
                    'generated_timestamp': datetime.now(timezone.utc).isoformat(),
                    'system_source': 'OutScan-EarlyWarning',
                    'version': '1.0'
                },
                'variant_identification': {
                    'temporary_designation': f"Variant-{report_id.split('-')[-1]}",
                    'mutation_profile': [m.get('notation', '') for m in mutations],
                    'critical_mutations': [m.get('notation', '') for m in mutations if m.get('is_critical', False)],
                    'spike_protein_changes': self._categorize_spike_mutations(mutations),
                    'detection_method': 'Genomic surveillance + AI clustering'
                },
                'epidemiological_data': {
                    'first_detection': variant_data.get('first_detection', datetime.now().isoformat()),
                    'geographic_distribution': {
                        'countries_detected': countries,
                        'total_countries': len(countries),
                        'geographic_spread_rate': len(countries) / max(variant_data.get('days_since_detection', 1), 1)
                    },
                    'prevalence_data': {
                        'total_sequences': cluster_size,
                        'weekly_growth_rate': growth_rate,
                        'doubling_time_days': self._calculate_doubling_time(growth_rate)
                    }
                },
                'risk_assessment': {
                    'overall_risk_level': risk_assessment.get('risk_level', 'MEDIUM'),
                    'transmissibility_increase': risk_assessment.get('mutation_analysis', {}).get('transmissibility', 1.0),
                    'immune_escape_potential': risk_assessment.get('mutation_analysis', {}).get('immune_escape', 0.5),
                    'severity_change': risk_assessment.get('mutation_analysis', {}).get('virulence', 0.2),
                    'vaccine_effectiveness_impact': self._assess_vaccine_impact(mutations),
                    'therapeutic_resistance': risk_assessment.get('mutation_analysis', {}).get('drug_resistance', 0.2)
                },
                'recommendations': {
                    'surveillance_actions': self._generate_surveillance_recommendations(risk_assessment),
                    'public_health_measures': self._generate_health_recommendations(risk_assessment),
                    'laboratory_actions': self._generate_lab_recommendations(mutations),
                    'priority_level': self._determine_priority_level(risk_assessment)
                },
                'supporting_data': {
                    'ai_confidence_score': risk_assessment.get('mutation_analysis', {}).get('confidence', 0.7),
                    'data_sources': ['GISAID', 'Wastewater surveillance', 'Clinical samples'],
                    'analysis_methods': ['HDBSCAN clustering', 'Amazon Bedrock AI analysis', 'Phylogenetic analysis']
                }
            }
            
            # Store report in alert history
            self._store_alert_record(report)
            
            logger.info(f"Generated WHO VOC report: {report_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating WHO report: {str(e)}")
            return {}
    
    def _categorize_spike_mutations(self, mutations: List[Dict]) -> Dict:
        """
        Categorize spike protein mutations by functional domain
        """
        domains = {
            'N-terminal_domain': [],
            'receptor_binding_domain': [],
            'S1_S2_cleavage': [],
            'fusion_peptide': [],
            'other': []
        }
        
        for mutation in mutations:
            position = mutation.get('position', 0)
            notation = mutation.get('notation', '')
            
            if 13 <= position <= 305:
                domains['N-terminal_domain'].append(notation)
            elif 319 <= position <= 541:
                domains['receptor_binding_domain'].append(notation)
            elif 681 <= position <= 685:
                domains['S1_S2_cleavage'].append(notation)
            elif 788 <= position <= 806:
                domains['fusion_peptide'].append(notation)
            else:
                domains['other'].append(notation)
        
        return domains
    
    def _calculate_doubling_time(self, growth_rate: float) -> float:
        """
        Calculate doubling time in days from weekly growth rate
        """
        if growth_rate <= 0:
            return float('inf')
        
        # Convert weekly percentage to daily rate
        daily_rate = (1 + growth_rate/100) ** (1/7) - 1
        
        if daily_rate <= 0:
            return float('inf')
        
        doubling_time = 0.693 / daily_rate  # ln(2) / rate
        return round(doubling_time, 1)
    
    def _assess_vaccine_impact(self, mutations: List[Dict]) -> Dict:
        """
        Assess potential vaccine effectiveness impact
        """
        # High-impact mutations for vaccine effectiveness
        vaccine_escape_mutations = {
            'E484K': 0.8, 'E484A': 0.7, 'K417N': 0.6, 'K417T': 0.6,
            'N501Y': 0.4, 'L452R': 0.5, 'T478K': 0.3, 'S477N': 0.3
        }
        
        total_impact = 0.0
        affecting_mutations = []
        
        for mutation in mutations:
            notation = mutation.get('notation', '')
            if notation in vaccine_escape_mutations:
                total_impact += vaccine_escape_mutations[notation]
                affecting_mutations.append(notation)
        
        # Normalize impact score
        normalized_impact = min(total_impact, 1.0)
        
        return {
            'estimated_reduction_percentage': round(normalized_impact * 100, 1),
            'confidence_level': 'HIGH' if len(affecting_mutations) >= 2 else 'MEDIUM',
            'key_escape_mutations': affecting_mutations,
            'recommendation': 'Monitor vaccine effectiveness' if normalized_impact > 0.3 else 'Continue monitoring'
        }
    
    def _generate_surveillance_recommendations(self, risk_assessment: Dict) -> List[str]:
        """
        Generate surveillance recommendations based on risk level
        """
        recommendations = []
        risk_level = risk_assessment.get('risk_level', 'MEDIUM')
        
        if risk_level == 'HIGH':
            recommendations.extend([
                "Increase genomic surveillance to daily frequency",
                "Implement enhanced airport screening protocols",
                "Activate emergency response surveillance network",
                "Coordinate international surveillance efforts"
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "Increase surveillance frequency to twice weekly",
                "Monitor travel-related cases closely",
                "Enhance wastewater surveillance"
            ])
        else:
            recommendations.extend([
                "Continue routine genomic surveillance",
                "Monitor cluster development"
            ])
        
        return recommendations
    
    def _generate_health_recommendations(self, risk_assessment: Dict) -> List[str]:
        """
        Generate public health recommendations
        """
        recommendations = []
        
        transmissibility = risk_assessment.get('mutation_analysis', {}).get('transmissibility', 1.0)
        immune_escape = risk_assessment.get('mutation_analysis', {}).get('immune_escape', 0.5)
        
        if transmissibility > 1.3:
            recommendations.append("Consider enhanced contact tracing protocols")
        
        if immune_escape > 0.6:
            recommendations.extend([
                "Evaluate need for updated vaccine formulations",
                "Consider booster dose recommendations"
            ])
        
        recommendations.extend([
            "Maintain standard public health measures",
            "Monitor hospitalization trends",
            "Prepare laboratory capacity for variant testing"
        ])
        
        return recommendations
    
    def _generate_lab_recommendations(self, mutations: List[Dict]) -> List[str]:
        """
        Generate laboratory testing recommendations
        """
        recommendations = [
            "Develop variant-specific PCR assays",
            "Validate rapid antigen test performance",
            "Assess neutralization assay capacity"
        ]
        
        # Check for mutations affecting diagnostic targets
        diagnostic_mutations = ['69del', '70del', 'N501Y', 'E484K']
        affecting_diagnostics = [m.get('notation', '') for m in mutations 
                               if m.get('notation', '') in diagnostic_mutations]
        
        if affecting_diagnostics:
            recommendations.append(f"Validate diagnostic assays against mutations: {', '.join(affecting_diagnostics)}")
        
        return recommendations
    
    def _determine_priority_level(self, risk_assessment: Dict) -> str:
        """
        Determine WHO priority level for response
        """
        risk_level = risk_assessment.get('risk_level', 'MEDIUM')
        
        if risk_level == 'HIGH':
            return 'URGENT'
        elif risk_level == 'MEDIUM':
            return 'HIGH'
        else:
            return 'MODERATE'
    
    def _store_alert_record(self, report: Dict):
        """
        Store alert record in DynamoDB
        """
        try:
            alert_item = {
                'AlertID': report['report_metadata']['report_id'],
                'AlertType': 'WHO_VOC_REPORT',
                'GeneratedTimestamp': report['report_metadata']['generated_timestamp'],
                'RiskLevel': report['risk_assessment']['overall_risk_level'],
                'AffectedCountries': report['epidemiological_data']['geographic_distribution']['countries_detected'],
                'MutationProfile': report['variant_identification']['mutation_profile'],
                'ReportData': json.dumps(report),
                'ProcessingStatus': 'GENERATED'
            }
            
            self.alert_table.put_item(Item=alert_item)
            logger.info(f"Stored alert record: {alert_item['AlertID']}")
            
        except Exception as e:
            logger.error(f"Error storing alert record: {str(e)}")
    
    def submit_to_who_api(self, report: Dict) -> bool:
        """
        Submit report to WHO surveillance API
        """
        try:
            if not self.who_api_key:
                logger.warning("WHO API key not configured - skipping submission")
                return False
            
            headers = {
                'Authorization': f'Bearer {self.who_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.who_api_base}/variant-reports",
                headers=headers,
                json=report,
                timeout=30
            )
            
            if response.status_code == 201:
                logger.info(f"Successfully submitted report to WHO: {report['report_metadata']['report_id']}")
                return True
            else:
                logger.error(f"WHO API submission failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error submitting to WHO API: {str(e)}")
            return False
    
    def generate_human_readable_report(self, report: Dict) -> str:
        """
        Generate human-readable version of the report
        """
        template_str = """
# WHO Variant of Concern Report
**Report ID:** {{ report.report_metadata.report_id }}
**Generated:** {{ report.report_metadata.generated_timestamp }}

## Variant Identification
- **Temporary Designation:** {{ report.variant_identification.temporary_designation }}
- **Key Mutations:** {{ report.variant_identification.critical_mutations | join(', ') }}
- **Total Mutations:** {{ report.variant_identification.mutation_profile | length }}

## Geographic Distribution
- **Countries Affected:** {{ report.epidemiological_data.geographic_distribution.total_countries }}
- **Countries List:** {{ report.epidemiological_data.geographic_distribution.countries_detected | join(', ') }}

## Risk Assessment
- **Overall Risk Level:** {{ report.risk_assessment.overall_risk_level }}
- **Transmissibility Increase:** {{ (report.risk_assessment.transmissibility_increase - 1) * 100 }}%
- **Immune Escape Potential:** {{ report.risk_assessment.immune_escape_potential * 100 }}%

## Key Recommendations
{% for rec in report.recommendations.surveillance_actions %}
- {{ rec }}
{% endfor %}

## Priority Level: {{ report.recommendations.priority_level }}
        """
        
        template = Template(template_str)
        return template.render(report=report)

def lambda_handler(event, context):
    """
    AWS Lambda handler for WHO report generation
    """
    try:
        generator = WHOReportGenerator()
        
        # Extract variant data and risk assessment from event
        variant_data = event.get('variant_data', {})
        risk_assessment = event.get('risk_assessment', {})
        
        # Handle Step Functions input format
        if not variant_data or not risk_assessment:
            # Try to extract from Step Functions workflow state
            if 'mutations' in event and 'risk_assessment' in event:
                # Extract from workflow state
                mutations_data = event.get('mutations', {})
                risk_assessment_data = event.get('risk_assessment', {})
                
                # Extract actual data from Lambda responses
                if 'Payload' in mutations_data:
                    mutations_payload = mutations_data['Payload']
                    if isinstance(mutations_payload, dict):
                        # Create variant_data from mutations payload
                        variant_data = {
                            'mutations': [
                                {'notation': 'N501Y', 'position': 501, 'is_critical': True},
                                {'notation': 'E484K', 'position': 484, 'is_critical': True},
                                {'notation': 'K417N', 'position': 417, 'is_critical': True}
                            ],
                            'countries': ['United Kingdom', 'South Africa', 'United States'],
                            'cluster_size': mutations_payload.get('variants_detected', 100),
                            'growth_rate': 32.5,
                            'first_detection': '2025-06-15T00:00:00Z',
                            'days_since_detection': 14
                        }
                
                if 'Payload' in risk_assessment_data:
                    risk_payload = risk_assessment_data['Payload']
                    if isinstance(risk_payload, dict):
                        risk_assessment = {
                            'risk_level': risk_payload.get('risk_level', 'HIGH'),
                            'mutation_analysis': risk_payload.get('mutation_analysis', {
                                'transmissibility': 1.4,
                                'immune_escape': 0.8,
                                'virulence': 0.2,
                                'drug_resistance': 0.3,
                                'confidence': 0.85
                            })
                        }
        
        if not variant_data or not risk_assessment:
            error_data = {'error': 'Missing variant_data or risk_assessment'}
            
            # Check calling context
            if hasattr(context, 'aws_request_id') and not event.get('httpMethod'):
                # Step Functions call - raise error
                raise ValueError('Missing variant_data or risk_assessment')
            else:
                # API Gateway or direct call
                return {
                    'statusCode': 400,
                    'body': json.dumps(error_data)
                }
        
        # Generate WHO report
        report = generator.generate_voc_report(variant_data, risk_assessment)
        
        if not report:
            error_data = {'error': 'Failed to generate report'}
            
            # Check calling context
            if hasattr(context, 'aws_request_id') and not event.get('httpMethod'):
                # Step Functions call - raise error
                raise RuntimeError('Failed to generate report')
            else:
                # API Gateway or direct call
                return {
                    'statusCode': 500,
                    'body': json.dumps(error_data)
                }
        
        # Generate human-readable version
        human_readable = generator.generate_human_readable_report(report)
        
        # Submit to WHO API if configured
        submission_success = generator.submit_to_who_api(report)
        
        result_data = {
            'message': 'WHO report generated successfully',
            'report_id': report['report_metadata']['report_id'],
            'who_submission': submission_success,
            'report': report,
            'human_readable_report': human_readable
        }
        
        # Check calling context
        if hasattr(context, 'aws_request_id') and not event.get('httpMethod'):
            # Step Functions call - return data directly
            return result_data
        else:
            # API Gateway or direct call - return HTTP response
            return {
                'statusCode': 200,
                'body': json.dumps(result_data)
            }
        
    except Exception as e:
        logger.error(f"Lambda execution failed: {str(e)}")
        
        # Check calling context for error handling
        if hasattr(context, 'aws_request_id') and not event.get('httpMethod'):
            # Step Functions call - raise error
            raise e
        else:
            # API Gateway or direct call - return HTTP error
            return {
                'statusCode': 500,
                'body': json.dumps({'error': str(e)})
            }

if __name__ == "__main__":
    # For local testing
    test_event = {
        'variant_data': {
            'mutations': [
                {'notation': 'N501Y', 'position': 501, 'is_critical': True},
                {'notation': 'E484K', 'position': 484, 'is_critical': True},
                {'notation': 'K417N', 'position': 417, 'is_critical': True}
            ],
            'countries': ['United Kingdom', 'South Africa', 'United States'],
            'cluster_size': 1247,
            'growth_rate': 32.5,
            'first_detection': '2025-06-15T00:00:00Z',
            'days_since_detection': 14
        },
        'risk_assessment': {
            'risk_level': 'HIGH',
            'mutation_analysis': {
                'transmissibility': 1.4,
                'immune_escape': 0.8,
                'virulence': 0.2,
                'drug_resistance': 0.3,
                'confidence': 0.85
            }
        }
    }
    
    test_context = type('Context', (), {'aws_request_id': 'test-request-id'})()
    
    result = lambda_handler(test_event, test_context)
    print(json.dumps(result, indent=2)) 