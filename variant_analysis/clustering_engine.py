"""
SARS-CoV-2 Variant Clustering Engine
HDBSCAN Implementation for Emergent Variant Detection

This module implements unsupervised machine learning to identify emerging 
variant clusters from spike protein sequences using HDBSCAN clustering 
and Jaccard distance matrices.

Author: OutScan Team
Version: 1.0
"""

import json
import boto3
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VariantClusteringEngine:
    """HDBSCAN-based clustering engine for variant detection"""
    
    def __init__(self):
        """Initialize the clustering engine with AWS services and parameters"""
        self.dynamodb = boto3.resource('dynamodb')
        self.variant_table = self.dynamodb.Table(os.environ.get('VARIANT_CLUSTERS_TABLE', 'VariantClusters'))
        self.mutation_table = self.dynamodb.Table(os.environ.get('MUTATION_LIBRARY_TABLE', 'MutationLibrary'))
        
        # Clustering parameters - configurable via environment variables
        self.min_cluster_size = int(os.environ.get('MIN_CLUSTER_SIZE', '10'))
        self.min_samples = int(os.environ.get('MIN_SAMPLES', '5'))
        self.cluster_selection_epsilon = float(os.environ.get('CLUSTER_SELECTION_EPSILON', '0.1'))
        
        # Known variant signatures for comparison
        self.known_variants = {
            'Alpha': ['N501Y', 'D614G', 'P681H'],
            'Beta': ['N501Y', 'E484K', 'K417N', 'D614G'],
            'Gamma': ['N501Y', 'E484K', 'K417T', 'D614G'], 
            'Delta': ['L452R', 'T478K', 'D614G', 'P681R'],
            'Omicron': ['G142D', 'K417N', 'N440K', 'G446S', 'S477N', 'T478K', 
                       'E484A', 'Q493R', 'G496S', 'Q498R', 'N501Y', 'Y505H']
        }
        
        logger.info(f"Clustering engine initialized with min_cluster_size={self.min_cluster_size}")

    def load_variant_data(self, days_back: int = 30) -> pd.DataFrame:
        """Load variant data from DynamoDB for clustering analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Loading variant data from {start_date.date()} to {end_date.date()}")
            
            # Scan DynamoDB for recent mutations
            response = self.mutation_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('upload_date').between(
                    start_date.isoformat(), end_date.isoformat()
                ),
                ProjectionExpression='sequence_id, spike_mutations, upload_date, geographic_location, collection_date, cluster_id'
            )
            
            variants = []
            for item in response['Items']:
                # Ensure spike_mutations is a list
                spike_mutations = item.get('spike_mutations', [])
                if isinstance(spike_mutations, str):
                    spike_mutations = spike_mutations.split(',')
                    
                variants.append({
                    'sequence_id': item['sequence_id'],
                    'spike_mutations': spike_mutations,
                    'upload_date': item['upload_date'],
                    'geographic_location': item.get('geographic_location', 'Unknown'),
                    'collection_date': item.get('collection_date', ''),
                    'existing_cluster': item.get('cluster_id', -1)
                })
            
            df = pd.DataFrame(variants)
            logger.info(f"Loaded {len(df)} variant sequences")
            return df
            
        except Exception as e:
            logger.error(f"Error loading from DynamoDB: {e}")
            raise

    def jaccard_distance(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard distance between two mutation sets"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return 1 - (intersection / union if union > 0 else 0)

    def calculate_distance_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate pairwise distance matrix for mutation signatures"""
        logger.info("Calculating pairwise distance matrix...")
        
        mutation_sets = [set(mutations) for mutations in df['spike_mutations']]
        n = len(mutation_sets)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.jaccard_distance(mutation_sets[i], mutation_sets[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        logger.info(f"Distance matrix calculated: {n}x{n}")
        return distance_matrix

    def compare_to_known_variants(self, mutation_set: Set[str]) -> Dict[str, float]:
        """Compare mutation set to known variant signatures"""
        similarities = {}
        for variant_name, variant_mutations in self.known_variants.items():
            variant_set = set(variant_mutations)
            jaccard_sim = 1 - self.jaccard_distance(mutation_set, variant_set)
            similarities[variant_name] = jaccard_sim
        return similarities

    def perform_clustering(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Perform HDBSCAN clustering on variant data"""
        logger.info("Starting HDBSCAN clustering analysis...")
        
        if len(df) < self.min_cluster_size:
            raise ValueError(f"Insufficient data for clustering: {len(df)} < {self.min_cluster_size}")
        
        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(df)
        
        # Perform HDBSCAN clustering
        logger.info("Performing HDBSCAN clustering...")
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric='precomputed'
        )
        
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # Add cluster labels to dataframe
        df_clustered = df.copy()
        df_clustered['cluster_id'] = cluster_labels
        df_clustered['cluster_probability'] = clusterer.probabilities_
        
        # Analyze clusters
        cluster_analysis = self.analyze_clusters(df_clustered, clusterer)
        
        logger.info(f"Clustering complete: {cluster_analysis['total_clusters']} clusters detected")
        return df_clustered, cluster_analysis

    def analyze_clusters(self, df_clustered: pd.DataFrame, clusterer) -> Dict:
        """Analyze detected clusters for emergent variants"""
        logger.info("Analyzing clusters for emergent variants...")
        
        analysis = {
            'total_clusters': len(set(df_clustered['cluster_id'])) - (1 if -1 in df_clustered['cluster_id'].values else 0),
            'noise_points': sum(df_clustered['cluster_id'] == -1),
            'cluster_details': {},
            'emergent_variants': [],
            'cluster_stability': clusterer.cluster_persistence_.tolist() if hasattr(clusterer, 'cluster_persistence_') else []
        }
        
        for cluster_id in sorted(df_clustered['cluster_id'].unique()):
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_data = df_clustered[df_clustered['cluster_id'] == cluster_id]
            
            # Find consensus mutations for this cluster
            all_mutations = []
            for mutations in cluster_data['spike_mutations']:
                all_mutations.extend(mutations)
            mutation_counts = Counter(all_mutations)
            
            # Mutations present in >50% of cluster members
            consensus_mutations = {mut for mut, count in mutation_counts.items() 
                                 if count > len(cluster_data) * 0.5}
            
            # Compare to known variants
            similarities = self.compare_to_known_variants(consensus_mutations)
            max_similarity = max(similarities.values()) if similarities else 0
            
            # Check if this is a potential emergent variant
            is_emergent = (
                max_similarity < 0.7 and  # Low similarity to known variants
                len(consensus_mutations) >= 3 and  # Has significant mutations
                len(cluster_data) >= self.min_cluster_size  # Sufficient cluster size
            )
            
            cluster_info = {
                'size': len(cluster_data),
                'consensus_mutations': list(consensus_mutations),
                'geographic_distribution': cluster_data['geographic_location'].value_counts().to_dict(),
                'temporal_span': {
                    'earliest': cluster_data['upload_date'].min(),
                    'latest': cluster_data['upload_date'].max()
                },
                'known_variant_similarities': similarities,
                'max_similarity_to_known': max_similarity,
                'is_emergent_variant': is_emergent,
                'growth_rate': self._calculate_growth_rate(cluster_data)
            }
            
            analysis['cluster_details'][cluster_id] = cluster_info
            
            if is_emergent:
                risk_score = self._calculate_variant_risk_score(cluster_info)
                analysis['emergent_variants'].append({
                    'cluster_id': cluster_id,
                    'consensus_mutations': list(consensus_mutations),
                    'size': len(cluster_data),
                    'geographic_spread': len(cluster_data['geographic_location'].unique()),
                    'risk_score': risk_score
                })
                
                logger.warning(f"Emergent variant detected in cluster {cluster_id}: risk_score={risk_score:.3f}")
        
        return analysis

    def _calculate_growth_rate(self, cluster_data: pd.DataFrame) -> float:
        """Calculate exponential growth rate for cluster"""
        try:
            dates = pd.to_datetime(cluster_data['upload_date']).sort_values()
            if len(dates) < 2:
                return 0.0
            
            # Simple exponential growth calculation
            days_span = (dates.iloc[-1] - dates.iloc[0]).days
            if days_span == 0:
                return 0.0
                
            return len(cluster_data) / max(days_span, 1)
        except Exception:
            return 0.0

    def _calculate_variant_risk_score(self, cluster_info: Dict) -> float:
        """Calculate risk score for potential emergent variant"""
        # Size factor (larger clusters are higher risk)
        size_score = min(cluster_info['size'] / 100, 1.0) * 0.3
        
        # Geographic spread factor
        geo_diversity = len(cluster_info['geographic_distribution'])
        geo_score = min(geo_diversity / 5, 1.0) * 0.3
        
        # Growth rate factor
        growth_score = min(cluster_info['growth_rate'] / 10, 1.0) * 0.2
        
        # Novelty factor (inverse of similarity to known variants)
        novelty_score = (1 - cluster_info['max_similarity_to_known']) * 0.2
        
        return size_score + geo_score + growth_score + novelty_score

    def save_clustering_results(self, df_clustered: pd.DataFrame, cluster_analysis: Dict) -> Dict:
        """Save clustering results back to DynamoDB"""
        try:
            logger.info("Saving clustering results to DynamoDB...")
            
            # Update sequence records with cluster assignments
            updated_sequences = 0
            with self.mutation_table.batch_writer() as batch:
                for _, row in df_clustered.iterrows():
                    batch.update_item(
                        Key={'sequence_id': row['sequence_id']},
                        UpdateExpression='SET cluster_id = :cid, cluster_probability = :prob, last_clustered = :ts',
                        ExpressionAttributeValues={
                            ':cid': int(row['cluster_id']),
                            ':prob': float(row['cluster_probability']),
                            ':ts': datetime.now().isoformat()
                        }
                    )
                    updated_sequences += 1
            
            # Save cluster metadata
            cluster_metadata = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_sequences_analyzed': len(df_clustered),
                'clusters_detected': cluster_analysis['total_clusters'],
                'emergent_variants_count': len(cluster_analysis['emergent_variants']),
                'analysis_parameters': {
                    'min_cluster_size': self.min_cluster_size,
                    'min_samples': self.min_samples,
                    'cluster_selection_epsilon': self.cluster_selection_epsilon
                }
            }
            
            # Store emergent variants for alerting pipeline
            emergent_variants_stored = 0
            for variant in cluster_analysis['emergent_variants']:
                self.variant_table.put_item(
                    Item={
                        'cluster_id': variant['cluster_id'],
                        'discovery_date': datetime.now().isoformat(),
                        'consensus_mutations': variant['consensus_mutations'],
                        'cluster_size': variant['size'],
                        'geographic_spread': variant['geographic_spread'],
                        'risk_score': variant['risk_score'],
                        'status': 'detected',
                        'alert_sent': False,
                        'metadata': cluster_metadata
                    }
                )
                emergent_variants_stored += 1
            
            logger.info(f"Results saved: {updated_sequences} sequences updated, {emergent_variants_stored} emergent variants stored")
            
            return {
                'success': True,
                'sequences_updated': updated_sequences,
                'emergent_variants_stored': emergent_variants_stored,
                'metadata': cluster_metadata
            }
            
        except Exception as e:
            logger.error(f"Error saving results to DynamoDB: {e}")
            return {
                'success': False,
                'error': str(e),
                'sequences_updated': 0,
                'emergent_variants_stored': 0
            }


def lambda_handler(event, context):
    """
    AWS Lambda function handler for clustering analysis
    
    Event structure:
    {
        "days_back": 30,  # Number of days to look back for variant data
        "trigger_alerts": true  # Whether to trigger alerts for emergent variants
    }
    """
    try:
        # Initialize clustering engine
        engine = VariantClusteringEngine()
        
        # Extract parameters from event
        days_back = event.get('days_back', 30)
        trigger_alerts = event.get('trigger_alerts', True)
        
        logger.info(f"Starting clustering analysis for last {days_back} days")
        
        # Load recent variant data
        df_variants = engine.load_variant_data(days_back)
        
        if len(df_variants) < engine.min_cluster_size:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Insufficient data for clustering (need >{engine.min_cluster_size}, got {len(df_variants)})',
                    'clusters_detected': 0,
                    'emergent_variants': [],
                    'analysis_timestamp': datetime.now().isoformat()
                })
            }
        
        # Perform clustering
        df_clustered, cluster_analysis = engine.perform_clustering(df_variants)
        
        # Save results
        save_result = engine.save_clustering_results(df_clustered, cluster_analysis)
        
        # Prepare response
        response_body = {
            'message': 'Clustering analysis completed successfully',
            'sequences_analyzed': len(df_clustered),
            'clusters_detected': cluster_analysis['total_clusters'],
            'emergent_variants': len(cluster_analysis['emergent_variants']),
            'emergent_variant_details': cluster_analysis['emergent_variants'],
            'save_result': save_result,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Trigger alerts for high-risk emergent variants
        if trigger_alerts:
            high_risk_variants = [v for v in cluster_analysis['emergent_variants'] if v['risk_score'] > 0.4]
            if high_risk_variants:
                logger.warning(f"{len(high_risk_variants)} high-risk emergent variants detected!")
                
                # Trigger SNS notifications (implement based on your alerting setup)
                try:
                    sns = boto3.client('sns')
                    topic_arn = os.environ.get('EMERGENT_VARIANT_TOPIC_ARN')
                    
                    if topic_arn:
                        for variant in high_risk_variants:
                            alert_message = {
                                'alert_type': 'emergent_variant',
                                'cluster_id': variant['cluster_id'],
                                'risk_score': variant['risk_score'],
                                'consensus_mutations': variant['consensus_mutations'],
                                'cluster_size': variant['size'],
                                'geographic_spread': variant['geographic_spread'],
                                'detection_timestamp': datetime.now().isoformat()
                            }
                            
                            sns.publish(
                                TopicArn=topic_arn,
                                Message=json.dumps(alert_message),
                                Subject=f"Emergent Variant Alert - Cluster {variant['cluster_id']}"
                            )
                            
                        logger.info(f"Alerts sent for {len(high_risk_variants)} high-risk variants")
                        
                except Exception as e:
                    logger.error(f"Error sending alerts: {e}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(response_body)
        }
        
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Clustering analysis failed',
                'details': str(e),
                'timestamp': datetime.now().isoformat()
            })
        }


# For local testing
if __name__ == "__main__":
    # Sample event for testing
    test_event = {
        'days_back': 30,
        'trigger_alerts': False
    }
    
    test_context = {}
    
    print("Testing clustering engine locally...")
    response = lambda_handler(test_event, test_context)
    print(json.dumps(response, indent=2)) 