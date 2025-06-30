"""
Simple SARS-CoV-2 Variant Clustering Engine
Pure Python implementation for variant detection without numpy/scipy dependencies

This module implements basic clustering to identify emerging variant clusters from 
mutation patterns using simple distance metrics and grouping algorithms.
"""

import json
import boto3
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SimpleClusteringEngine:
    """Pure Python clustering engine for variant detection"""
    
    def __init__(self):
        """Initialize the clustering engine with AWS services and parameters"""
        self.dynamodb = boto3.resource('dynamodb')
        
        # Use the correct table names that match the Lambda function expectations
        self.variant_table = self.dynamodb.Table('OutScan-VariantTable')
        self.sequence_table = self.dynamodb.Table('OutScan-SequenceTable')
        
        # Clustering parameters
        self.min_cluster_size = int(os.environ.get('MIN_CLUSTER_SIZE', '3'))
        self.distance_threshold = float(os.environ.get('CLUSTER_SELECTION_EPSILON', '0.6'))
        
        # Known variant signatures for comparison
        self.known_variants = {
            'Alpha': ['N501Y', 'D614G', 'P681H'],
            'Beta': ['N501Y', 'E484K', 'K417N', 'D614G'],
            'Gamma': ['N501Y', 'E484K', 'K417T', 'D614G'], 
            'Delta': ['L452R', 'T478K', 'D614G', 'P681R'],
            'Omicron': ['G142D', 'K417N', 'N440K', 'G446S', 'S477N', 'T478K', 
                       'E484A', 'Q493R', 'G496S', 'Q498R', 'N501Y', 'Y505H']
        }
        
        logger.info(f"Simple clustering engine initialized with min_cluster_size={self.min_cluster_size}")

    def load_variant_data(self) -> List[Dict]:
        """Load variant data from DynamoDB for clustering analysis"""
        try:
            logger.info("Loading variant data from DynamoDB...")
            
            # Scan variant table for all variants
            response = self.variant_table.scan(Limit=100)  # Limit to prevent timeouts
            
            variants = []
            for item in response['Items']:
                
                # Extract mutations from metadata if available
                mutations = []
                metadata = item.get('metadata', '{}')
                if isinstance(metadata, str):
                    try:
                        metadata_dict = json.loads(metadata)
                        if 'variant' in metadata_dict:
                            mutations = [metadata_dict['variant']]
                    except:
                        pass
                
                variants.append({
                    'variant_id': item['variant_id'],
                    'sequence_id': item['sequence_id'],
                    'variant_type': item.get('variant_type', 'unknown'),
                    'confidence': float(item.get('confidence', 0.5)),
                    'mutations': mutations,
                    'timestamp': item.get('timestamp', '')
                })
            
            logger.info(f"Loaded {len(variants)} variant records")
            return variants
            
        except Exception as e:
            logger.error(f"Error loading variant data: {e}")
            return []

    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two mutation sets"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def calculate_pairwise_distances(self, variants: List[Dict]) -> List[List[float]]:
        """Calculate pairwise distances between variants"""
        n = len(variants)
        distances = [[0.0 for _ in range(n)] for _ in range(n)]
        
        mutation_sets = [set(variant['mutations']) for variant in variants]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Use Jaccard distance (1 - similarity)
                similarity = self.jaccard_similarity(mutation_sets[i], mutation_sets[j])
                distance = 1.0 - similarity
                distances[i][j] = distance
                distances[j][i] = distance
        
        return distances

    def simple_clustering(self, variants: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Perform simple clustering using distance-based grouping"""
        logger.info("Starting simple clustering analysis...")
        
        if len(variants) < self.min_cluster_size:
            logger.warning(f"Insufficient data for clustering: {len(variants)} < {self.min_cluster_size}")
            return variants, {'total_clusters': 0, 'cluster_details': {}}
        
        # Calculate pairwise distances
        distances = self.calculate_pairwise_distances(variants)
        n = len(variants)
        
        # Simple clustering: group variants with distance < threshold
        clusters = []
        assigned = [False] * n
        
        for i in range(n):
            if assigned[i]:
                continue
                
            # Start new cluster
            cluster = [i]
            assigned[i] = True
            
            # Find all variants within threshold distance
            for j in range(i + 1, n):
                if not assigned[j] and distances[i][j] < self.distance_threshold:
                    cluster.append(j)
                    assigned[j] = True
            
            # Only keep clusters of minimum size
            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)
        
        # Assign cluster IDs to variants
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                variants[idx]['cluster_id'] = cluster_id
                variants[idx]['cluster_probability'] = 1.0
        
        # Mark unclustered variants
        for i, variant in enumerate(variants):
            if 'cluster_id' not in variant:
                variant['cluster_id'] = -1
                variant['cluster_probability'] = 0.0
        
        # Analyze clusters
        cluster_analysis = self.analyze_clusters(variants)
        
        logger.info(f"Clustering complete: {cluster_analysis['total_clusters']} clusters detected")
        return variants, cluster_analysis

    def analyze_clusters(self, variants: List[Dict]) -> Dict:
        """Analyze detected clusters for emergent variants"""
        logger.info("Analyzing clusters...")
        
        # Group variants by cluster
        clusters = defaultdict(list)
        for variant in variants:
            cluster_id = variant.get('cluster_id', -1)
            clusters[cluster_id].append(variant)
        
        analysis = {
            'total_clusters': len([cid for cid in clusters.keys() if cid >= 0]),
            'cluster_details': {},
            'emergent_variants': []
        }
        
        for cluster_id, cluster_variants in clusters.items():
            if cluster_id < 0:
                continue
                
            # Analyze cluster composition
            variant_types = Counter(v['variant_type'] for v in cluster_variants)
            confidence_scores = [v['confidence'] for v in cluster_variants]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            
            # Find dominant variant type
            dominant_type = variant_types.most_common(1)[0][0] if variant_types else 'unknown'
            
            # Calculate risk score based on cluster size and confidence
            risk_score = min(1.0, (len(cluster_variants) / 10.0) * avg_confidence)
            
            cluster_detail = {
                'cluster_id': cluster_id,
                'size': len(cluster_variants),
                'dominant_variant': dominant_type,
                'variant_composition': dict(variant_types),
                'average_confidence': round(avg_confidence, 3),
                'risk_score': round(risk_score, 3),
                'sequences': [v['sequence_id'] for v in cluster_variants]
            }
            
            analysis['cluster_details'][str(cluster_id)] = cluster_detail
            
            # Flag as emergent if high risk
            if risk_score > 0.7 and len(cluster_variants) >= self.min_cluster_size:
                analysis['emergent_variants'].append({
                    'cluster_id': cluster_id,
                    'variant_type': dominant_type,
                    'size': len(cluster_variants),
                    'risk_score': risk_score,
                    'confidence': avg_confidence
                })
        
        return analysis

    def save_clustering_results(self, variants: List[Dict], cluster_analysis: Dict) -> Dict:
        """Save clustering results back to DynamoDB"""
        logger.info("Saving clustering results...")
        
        results_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_variants_analyzed': len(variants),
            'total_clusters': cluster_analysis['total_clusters'],
            'emergent_variants': len(cluster_analysis['emergent_variants']),
            'cluster_details': cluster_analysis['cluster_details']
        }
        
        # For this simple implementation, we'll just log the results
        # In production, you might save to a separate clustering results table
        logger.info(f"Clustering analysis complete: {json.dumps(results_summary, indent=2)}")
        
        return results_summary

def lambda_handler(event, context):
    """Lambda handler for clustering analysis"""
    try:
        logger.info("Starting variant clustering analysis...")
        
        # Initialize clustering engine
        engine = SimpleClusteringEngine()
        
        # Load variant data
        variants = engine.load_variant_data()
        
        if not variants:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'No variant data available for clustering',
                    'total_variants': 0,
                    'total_clusters': 0
                })
            }
        
        # Perform clustering
        clustered_variants, cluster_analysis = engine.simple_clustering(variants)
        
        # Save results
        results = engine.save_clustering_results(clustered_variants, cluster_analysis)
        
        # Return summary
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Clustering analysis completed successfully',
                'total_variants': len(variants),
                'total_clusters': cluster_analysis['total_clusters'],
                'emergent_variants': len(cluster_analysis['emergent_variants']),
                'results': results
            })
        }
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Clustering analysis failed',
                'message': str(e)
            })
        } 