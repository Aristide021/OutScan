"""
Historical Variant Simulator
Validates OutScan system against known Alpha/Delta/Omicron timelines
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import boto3
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HistoricalVariantSimulator:
    """
    Simulate historical variant emergence to validate early detection capability
    """
    
    def __init__(self):
        self.historical_variants = self._load_historical_data()
        self.detection_results = []
    
    def _load_historical_data(self) -> Dict:
        """
        Load historical variant data for simulation
        """
        return {
            'Alpha': {
                'first_sequence': '2020-09-20',
                'who_voc_date': '2020-12-18',
                'peak_dominance': '2021-04-15',
                'key_mutations': ['N501Y', '69del', '70del', 'P681H'],
                'countries_timeline': {
                    '2020-09-20': ['United Kingdom'],
                    '2020-11-15': ['United Kingdom', 'Denmark'],
                    '2020-12-01': ['United Kingdom', 'Denmark', 'Netherlands', 'Belgium'],
                    '2020-12-15': ['United Kingdom', 'Denmark', 'Netherlands', 'Belgium', 'Germany', 'France'],
                    '2021-01-01': ['United Kingdom', 'Denmark', 'Netherlands', 'Belgium', 'Germany', 'France', 'United States', 'Canada']
                },
                'growth_rates': {
                    '2020-10-01': 15.2,
                    '2020-11-01': 28.5,
                    '2020-12-01': 45.8,
                    '2021-01-01': 32.1
                }
            },
            'Delta': {
                'first_sequence': '2020-10-11',
                'who_voc_date': '2021-05-11',
                'peak_dominance': '2021-08-15',
                'key_mutations': ['L452R', 'T478K', 'P681R', 'T19R'],
                'countries_timeline': {
                    '2020-10-11': ['India'],
                    '2021-02-01': ['India', 'United Kingdom'],
                    '2021-04-01': ['India', 'United Kingdom', 'United States', 'Singapore'],
                    '2021-05-01': ['India', 'United Kingdom', 'United States', 'Singapore', 'Germany', 'Canada'],
                    '2021-06-01': ['India', 'United Kingdom', 'United States', 'Singapore', 'Germany', 'Canada', 'Australia', 'Japan']
                },
                'growth_rates': {
                    '2021-01-01': 8.5,
                    '2021-03-01': 42.3,
                    '2021-04-01': 67.8,
                    '2021-05-01': 58.2
                }
            },
            'Omicron': {
                'first_sequence': '2021-11-09',
                'who_voc_date': '2021-11-26',
                'peak_dominance': '2022-01-15',
                'key_mutations': ['G339D', 'S371L', 'S373P', 'S375F', 'K417N', 'N440K', 'G446S', 'S477N', 'T478K', 'E484A', 'Q493R', 'G496S', 'Q498R', 'N501Y', 'Y505H'],
                'countries_timeline': {
                    '2021-11-09': ['South Africa'],
                    '2021-11-20': ['South Africa', 'Botswana', 'Hong Kong'],
                    '2021-11-26': ['South Africa', 'Botswana', 'Hong Kong', 'United Kingdom', 'Netherlands'],
                    '2021-12-01': ['South Africa', 'Botswana', 'Hong Kong', 'United Kingdom', 'Netherlands', 'Germany', 'United States', 'Canada'],
                    '2021-12-15': ['South Africa', 'Botswana', 'Hong Kong', 'United Kingdom', 'Netherlands', 'Germany', 'United States', 'Canada', 'Australia', 'France', 'Denmark', 'Norway']
                },
                'growth_rates': {
                    '2021-11-15': 95.2,
                    '2021-12-01': 125.8,
                    '2021-12-15': 89.3,
                    '2022-01-01': 67.1
                }
            }
        }
    
    def simulate_variant_emergence(self, variant_name: str, 
                                  start_date: str, 
                                  detection_threshold: float = 0.7) -> Dict:
        """
        Simulate variant emergence and test OutScan detection timeline
        """
        try:
            variant_data = self.historical_variants[variant_name]
            simulation_results = {
                'variant_name': variant_name,
                'simulation_start': start_date,
                'historical_who_date': variant_data['who_voc_date'],
                'detection_timeline': [],
                'lead_time_achieved': 0,
                'detection_accuracy': {}
            }
            
            # Simulate day-by-day emergence
            current_date = datetime.strptime(start_date, '%Y-%m-%d')
            who_date = datetime.strptime(variant_data['who_voc_date'], '%Y-%m-%d')
            
            while current_date <= who_date:
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Get geographic spread for this date
                countries = self._get_countries_for_date(variant_data, date_str)
                growth_rate = self._get_growth_rate_for_date(variant_data, date_str)
                
                # Simulate OutScan risk assessment
                risk_score = self._calculate_simulated_risk(
                    variant_data['key_mutations'],
                    len(countries),
                    growth_rate
                )
                
                # Record detection event
                detection_event = {
                    'date': date_str,
                    'countries_detected': len(countries),
                    'growth_rate': growth_rate,
                    'risk_score': risk_score,
                    'alert_triggered': risk_score >= detection_threshold,
                    'days_before_who': (who_date - current_date).days
                }
                
                simulation_results['detection_timeline'].append(detection_event)
                
                # Check if first detection occurred
                if risk_score >= detection_threshold and simulation_results['lead_time_achieved'] == 0:
                    simulation_results['lead_time_achieved'] = (who_date - current_date).days
                    logger.info(f"OutScan would detect {variant_name} {simulation_results['lead_time_achieved']} days before WHO VOC designation")
                
                current_date += timedelta(days=7)  # Weekly simulation
            
            # Calculate accuracy metrics
            simulation_results['detection_accuracy'] = self._calculate_accuracy_metrics(
                simulation_results['detection_timeline'],
                variant_data
            )
            
            return simulation_results
            
        except Exception as e:
            logger.error(f"Error simulating {variant_name}: {str(e)}")
            return {}
    
    def _get_countries_for_date(self, variant_data: Dict, target_date: str) -> List[str]:
        """
        Get countries where variant was detected by target date
        """
        countries = []
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        for date_str, country_list in variant_data['countries_timeline'].items():
            date_dt = datetime.strptime(date_str, '%Y-%m-%d')
            if date_dt <= target_dt:
                countries = country_list
        
        return countries
    
    def _get_growth_rate_for_date(self, variant_data: Dict, target_date: str) -> float:
        """
        Get growth rate for target date (interpolated if needed)
        """
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        growth_rates = variant_data['growth_rates']
        
        # Find closest date
        closest_date = None
        closest_diff = float('inf')
        
        for date_str in growth_rates.keys():
            date_dt = datetime.strptime(date_str, '%Y-%m-%d')
            diff = abs((target_dt - date_dt).days)
            if diff < closest_diff:
                closest_diff = diff
                closest_date = date_str
        
        return growth_rates.get(closest_date, 0.0)
    
    def _calculate_simulated_risk(self, mutations: List[str], 
                                country_count: int, growth_rate: float) -> float:
        """
        Simulate OutScan risk assessment algorithm
        """
        # Mutation risk component
        critical_mutations = ['N501Y', 'E484K', 'K417N', 'L452R', 'T478K', 'P681H', 'P681R']
        mutation_risk = len([m for m in mutations if m in critical_mutations]) / len(critical_mutations)
        
        # Geographic spread component
        spread_risk = min(country_count / 8.0, 1.0)  # Max risk at 8+ countries
        
        # Growth rate component  
        growth_risk = min(growth_rate / 50.0, 1.0)  # Max risk at 50%+ weekly growth
        
        # Weighted composite (matching OutScan algorithm)
        composite_risk = 0.4 * mutation_risk + 0.3 * spread_risk + 0.3 * growth_risk
        
        return composite_risk
    
    def _calculate_accuracy_metrics(self, timeline: List[Dict], variant_data: Dict) -> Dict:
        """
        Calculate detection accuracy metrics
        """
        # Count true positives, false positives, etc.
        detected_events = [event for event in timeline if event['alert_triggered']]
        
        # For historical variants, we know they were all significant
        true_positives = len(detected_events)
        false_negatives = 1 if len(detected_events) == 0 else 0
        
        return {
            'detection_rate': 1.0 if true_positives > 0 else 0.0,
            'average_lead_time': np.mean([event['days_before_who'] for event in detected_events]) if detected_events else 0,
            'peak_risk_score': max([event['risk_score'] for event in timeline]),
            'detection_sensitivity': true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        }
    
    def run_full_validation(self) -> Dict:
        """
        Run validation against all historical variants
        """
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'variants_tested': [],
            'overall_metrics': {},
            'individual_results': {}
        }
        
        lead_times = []
        detection_rates = []
        
        for variant_name, variant_data in self.historical_variants.items():
            logger.info(f"Simulating {variant_name} variant emergence...")
            
            # Start simulation 60 days before first sequence
            start_date = (datetime.strptime(variant_data['first_sequence'], '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')
            
            result = self.simulate_variant_emergence(variant_name, start_date)
            
            if result:
                validation_results['variants_tested'].append(variant_name)
                validation_results['individual_results'][variant_name] = result
                
                lead_times.append(result['lead_time_achieved'])
                detection_rates.append(result['detection_accuracy']['detection_rate'])
        
        # Calculate overall metrics
        validation_results['overall_metrics'] = {
            'average_lead_time_days': np.mean(lead_times) if lead_times else 0,
            'detection_success_rate': np.mean(detection_rates) if detection_rates else 0,
            'variants_detected_early': sum(1 for lt in lead_times if lt > 14),  # >2 weeks early
            'total_variants_tested': len(lead_times)
        }
        
        return validation_results
    
    def generate_validation_report(self, results: Dict) -> str:
        """
        Generate human-readable validation report
        """
        report = f"""
# OutScan Historical Validation Report
Generated: {results['validation_timestamp']}

## Overall Performance
- **Average Lead Time**: {results['overall_metrics']['average_lead_time_days']:.1f} days
- **Detection Success Rate**: {results['overall_metrics']['detection_success_rate']:.1%}
- **Variants Detected Early (>14 days)**: {results['overall_metrics']['variants_detected_early']}/{results['overall_metrics']['total_variants_tested']}

## Individual Variant Results
"""
        
        for variant_name, result in results['individual_results'].items():
            report += f"""
### {variant_name} Variant
- **Lead Time Achieved**: {result['lead_time_achieved']} days before WHO designation
- **Peak Risk Score**: {result['detection_accuracy']['peak_risk_score']:.2f}
- **Detection Rate**: {result['detection_accuracy']['detection_rate']:.1%}
- **Historical WHO Date**: {result['historical_who_date']}
"""
        
        return report

def lambda_handler(event, context):
    """
    AWS Lambda handler for historical validation
    """
    try:
        simulator = HistoricalVariantSimulator()
        
        # Run validation
        results = simulator.run_full_validation()
        
        # Generate report
        report = simulator.generate_validation_report(results)
        
        # Store results in S3
        s3_client = boto3.client('s3')
        s3_key = f"validation-results/{datetime.now().strftime('%Y-%m-%d')}/historical_validation.json"
        
        s3_client.put_object(
            Bucket='outscan-analysis-results',
            Key=s3_key,
            Body=json.dumps(results, indent=2),
            ContentType='application/json'
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Historical validation completed',
                'average_lead_time': results['overall_metrics']['average_lead_time_days'],
                'detection_success_rate': results['overall_metrics']['detection_success_rate'],
                'report_location': f's3://outscan-analysis-results/{s3_key}',
                'summary_report': report
            })
        }
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

if __name__ == "__main__":
    # Run local validation
    simulator = HistoricalVariantSimulator()
    results = simulator.run_full_validation()
    
    print("=== OUTSCAN HISTORICAL VALIDATION ===")
    print(f"Average Lead Time: {results['overall_metrics']['average_lead_time_days']:.1f} days")
    print(f"Detection Success Rate: {results['overall_metrics']['detection_success_rate']:.1%}")
    print(f"Early Detection Count: {results['overall_metrics']['variants_detected_early']}/{results['overall_metrics']['total_variants_tested']}")
    
    # Print detailed results
    for variant_name, result in results['individual_results'].items():
        print(f"\n{variant_name}: {result['lead_time_achieved']} days lead time") 