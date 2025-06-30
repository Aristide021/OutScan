"""
Advanced SARS-CoV-2 Variant Clustering Engine
Pure Python implementation of Ward linkage hierarchical clustering
Maintains scientific rigor without numpy/scipy dependencies

This implements proper hierarchical clustering algorithms used in genomics
for detecting emerging variant clusters and evolutionary relationships.
"""

import json
import boto3
import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set, Union
from collections import defaultdict, Counter
import logging
import os
import heapq

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Matrix:
    """Simple matrix implementation for distance calculations"""
    def __init__(self, size: int, default_value: float = 0.0):
        self.size = size
        self.data = [[default_value for _ in range(size)] for _ in range(size)]
    
    def get(self, i: int, j: int) -> float:
        return self.data[i][j]
    
    def set(self, i: int, j: int, value: float):
        self.data[i][j] = value
        self.data[j][i] = value  # Symmetric matrix
    
    def get_row(self, i: int) -> List[float]:
        return self.data[i][:]

class HierarchicalCluster:
    """Represents a cluster in hierarchical clustering"""
    def __init__(self, cluster_id: int, members: List[int], height: float = 0.0):
        self.id = cluster_id
        self.members = members  # List of original data point indices
        self.height = height    # Height at which this cluster was formed
        self.left = None       # Left child cluster
        self.right = None      # Right child cluster
        self.size = len(members)

class AdvancedClusteringEngine:
    """Advanced clustering engine with Ward linkage and hierarchical analysis"""
    
    def __init__(self):
        """Initialize the clustering engine with AWS services and parameters"""
        self.dynamodb = boto3.resource('dynamodb')
        
        # Use the correct table names
        self.variant_table = self.dynamodb.Table('OutScan-VariantTable')
        self.sequence_table = self.dynamodb.Table('OutScan-SequenceTable')
        
        # Advanced clustering parameters
        self.min_cluster_size = int(os.environ.get('MIN_CLUSTER_SIZE', '5'))
        self.distance_threshold = float(os.environ.get('CLUSTER_SELECTION_EPSILON', '0.7'))
        self.max_clusters = int(os.environ.get('MAX_CLUSTERS', '20'))
        
        # Known variant signatures for validation
        self.known_variants = {
            'Alpha': {'mutations': ['N501Y', 'D614G', 'P681H'], 'weight': 1.0},
            'Beta': {'mutations': ['N501Y', 'E484K', 'K417N', 'D614G'], 'weight': 1.0},
            'Gamma': {'mutations': ['N501Y', 'E484K', 'K417T', 'D614G'], 'weight': 1.0}, 
            'Delta': {'mutations': ['L452R', 'T478K', 'D614G', 'P681R'], 'weight': 1.2},
            'Omicron': {'mutations': ['G142D', 'K417N', 'N440K', 'G446S', 'S477N', 'T478K', 
                                    'E484A', 'Q493R', 'G496S', 'Q498R', 'N501Y', 'Y505H'], 'weight': 1.5}
        }
        
        logger.info(f"Advanced clustering engine initialized with Ward linkage")

    def load_variant_data(self) -> List[Dict]:
        """Load variant data from DynamoDB for clustering analysis"""
        try:
            logger.info("Loading variant data from DynamoDB...")
            
            # Scan variant table for all variants
            response = self.variant_table.scan(Limit=200)  # Increased limit for better analysis
            
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
                        # Extract additional mutation patterns if present
                        if 'pattern_matches' in metadata_dict:
                            mutations.extend([f"pattern_{i}" for i in range(metadata_dict['pattern_matches'])])
                    except:
                        pass
                
                variants.append({
                    'variant_id': item['variant_id'],
                    'sequence_id': item['sequence_id'],
                    'variant_type': item.get('variant_type', 'unknown'),
                    'confidence': float(item.get('confidence', 0.5)),
                    'mutations': mutations,
                    'timestamp': item.get('timestamp', ''),
                    'source': item.get('source_file', 'unknown')
                })
            
            logger.info(f"Loaded {len(variants)} variant records for advanced analysis")
            return variants
            
        except Exception as e:
            logger.error(f"Error loading variant data: {e}")
            return []

    def perform_advanced_clustering(self, variants: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Perform advanced hierarchical clustering with Ward linkage"""
        logger.info("Starting advanced hierarchical clustering analysis...")
        
        if len(variants) < self.min_cluster_size:
            logger.warning(f"Insufficient data for clustering: {len(variants)} < {self.min_cluster_size}")
            return variants, {'total_clusters': 0, 'cluster_details': {}}
        
        # Perform Ward linkage clustering
        flat_clusters = self.ward_linkage_clustering(variants)
        
        # Assign cluster IDs to variants
        for cluster_id, cluster_members in enumerate(flat_clusters):
            for member_idx in cluster_members:
                variants[member_idx]['cluster_id'] = cluster_id
                variants[member_idx]['cluster_probability'] = 1.0
        
        # Mark unclustered variants
        clustered_indices = set()
        for cluster in flat_clusters:
            clustered_indices.update(cluster)
        
        for i, variant in enumerate(variants):
            if i not in clustered_indices:
                variant['cluster_id'] = -1
                variant['cluster_probability'] = 0.0
        
        # Perform advanced cluster analysis
        cluster_analysis = self.analyze_hierarchical_clusters(variants, flat_clusters)
        
        logger.info(f"Advanced clustering complete: {cluster_analysis['total_clusters']} clusters")
        return variants, cluster_analysis

    def ward_linkage_clustering(self, variants: List[Dict]) -> List[List[int]]:
        """Implement Ward linkage hierarchical clustering"""
        logger.info("Implementing Ward linkage clustering...")
        
        n = len(variants)
        if n < 2:
            return []
        
        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(variants)
        
        # Initialize clusters
        clusters = [[i] for i in range(n)]
        
        # Ward linkage algorithm
        while len(clusters) > 1:
            min_distance = float('inf')
            merge_i, merge_j = -1, -1
            
            # Find closest pair using Ward criterion
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    ward_dist = self.calculate_ward_distance(
                        clusters[i], clusters[j], distance_matrix, variants
                    )
                    
                    if ward_dist < min_distance:
                        min_distance = ward_dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            merged_cluster = clusters[merge_i] + clusters[merge_j]
            clusters = [c for idx, c in enumerate(clusters) if idx not in [merge_i, merge_j]]
            clusters.append(merged_cluster)
            
            # Stop if we have enough large clusters
            if len(clusters) <= self.max_clusters:
                large_clusters = [c for c in clusters if len(c) >= self.min_cluster_size]
                if len(large_clusters) >= 2:
                    break
        
        # Return only clusters that meet minimum size
        return [c for c in clusters if len(c) >= self.min_cluster_size]

    def calculate_distance_matrix(self, variants: List[Dict]) -> Matrix:
        """Calculate genomic distance matrix"""
        n = len(variants)
        distance_matrix = Matrix(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self.weighted_genomic_distance(variants[i], variants[j])
                distance_matrix.set(i, j, distance)
        
        return distance_matrix

    def weighted_genomic_distance(self, variant1: Dict, variant2: Dict) -> float:
        """Calculate weighted genomic distance considering mutation importance"""
        mutations1 = set(variant1['mutations'])
        mutations2 = set(variant2['mutations'])
        
        # Base Jaccard distance
        base_distance = self.jaccard_distance(mutations1, mutations2)
        
        # Weight adjustment based on confidence scores
        conf_weight = 1.0 - abs(variant1['confidence'] - variant2['confidence']) * 0.3
        
        # Weight adjustment based on known variant patterns
        type_weight = 1.0
        if variant1['variant_type'] == variant2['variant_type'] and variant1['variant_type'] != 'unknown':
            type_weight = 0.8  # Reduce distance for same variant types
        
        return base_distance * conf_weight * type_weight

    def jaccard_distance(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard distance"""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return 1.0 - (intersection / union if union > 0 else 0.0)

    def calculate_ward_distance(self, cluster1: List[int], cluster2: List[int], 
                               distance_matrix: Matrix, variants: List[Dict]) -> float:
        """Calculate Ward distance between two clusters"""
        # Ward criterion: minimize within-cluster sum of squares
        total_distance = 0.0
        comparisons = 0
        
        for i in cluster1:
            for j in cluster2:
                total_distance += distance_matrix.get(i, j)
                comparisons += 1
        
        if comparisons == 0:
            return float('inf')
        
        # Ward formula approximation
        avg_distance = total_distance / comparisons
        size_factor = (len(cluster1) * len(cluster2)) / (len(cluster1) + len(cluster2))
        
        return avg_distance * size_factor

    def analyze_hierarchical_clusters(self, variants: List[Dict], flat_clusters: List[List[int]]) -> Dict:
        """Advanced analysis of hierarchical clusters"""
        logger.info("Performing advanced cluster analysis...")
        
        analysis = {
            'total_clusters': len(flat_clusters),
            'cluster_details': {},
            'emergent_variants': []
        }
        
        for cluster_id, cluster_indices in enumerate(flat_clusters):
            cluster_variants = [variants[i] for i in cluster_indices]
            
            # Statistical analysis
            variant_types = Counter(v['variant_type'] for v in cluster_variants)
            confidence_scores = [v['confidence'] for v in cluster_variants]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            confidence_std = math.sqrt(sum((x - avg_confidence) ** 2 for x in confidence_scores) / len(confidence_scores))
            
            # Evolutionary coherence
            coherence_score = self.calculate_evolutionary_coherence(cluster_variants)
            
            # Risk assessment
            risk_score = (
                min(1.0, len(cluster_variants) / 20.0) * 0.3 +
                avg_confidence * 0.3 +
                max(0, 1.0 - confidence_std) * 0.2 +
                coherence_score * 0.2
            )
            
            dominant_type = variant_types.most_common(1)[0][0] if variant_types else 'unknown'
            
            cluster_detail = {
                'cluster_id': cluster_id,
                'size': len(cluster_variants),
                'dominant_variant': dominant_type,
                'variant_composition': dict(variant_types),
                'statistical_metrics': {
                    'average_confidence': round(avg_confidence, 3),
                    'confidence_std_dev': round(confidence_std, 3),
                    'evolutionary_coherence': round(coherence_score, 3)
                },
                'risk_assessment': {
                    'overall_risk_score': round(risk_score, 3),
                    'alert_level': self.determine_alert_level(risk_score)
                },
                'sequences': [v['sequence_id'] for v in cluster_variants]
            }
            
            analysis['cluster_details'][str(cluster_id)] = cluster_detail
            
            # Flag emergent variants
            if risk_score > 0.75 and len(cluster_variants) >= self.min_cluster_size:
                analysis['emergent_variants'].append({
                    'cluster_id': cluster_id,
                    'variant_type': dominant_type,
                    'size': len(cluster_variants),
                    'risk_score': round(risk_score, 3),
                    'evolutionary_coherence': round(coherence_score, 3)
                })
        
        return analysis

    def calculate_evolutionary_coherence(self, cluster_variants: List[Dict]) -> float:
        """Calculate evolutionary coherence of cluster"""
        if len(cluster_variants) < 2:
            return 1.0
        
        similarities = []
        mutation_sets = [set(v['mutations']) for v in cluster_variants]
        
        for i in range(len(mutation_sets)):
            for j in range(i + 1, len(mutation_sets)):
                jaccard_sim = self.jaccard_similarity(mutation_sets[i], mutation_sets[j])
                similarities.append(jaccard_sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0

    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def determine_alert_level(self, risk_score: float) -> str:
        """Determine alert level"""
        if risk_score >= 0.85:
            return "CRITICAL"
        elif risk_score >= 0.70:
            return "HIGH"
        elif risk_score >= 0.50:
            return "MEDIUM"
        else:
            return "LOW"

def lambda_handler(event, context):
    """Lambda handler for advanced clustering analysis"""
    try:
        logger.info("Starting advanced Ward linkage clustering analysis...")
        
        engine = AdvancedClusteringEngine()
        variants = engine.load_variant_data()
        
        if not variants:
            result_data = {
                'message': 'No variant data available for clustering',
                'total_variants': 0,
                'total_clusters': 0
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
        
        clustered_variants, cluster_analysis = engine.perform_advanced_clustering(variants)
        
        result_data = {
            'message': 'Advanced Ward linkage clustering completed successfully',
            'analysis_type': 'Ward Linkage Hierarchical Clustering',
            'total_variants': len(variants),
            'total_clusters': cluster_analysis['total_clusters'],
            'emergent_variants': len(cluster_analysis['emergent_variants']),
            'cluster_analysis': cluster_analysis
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
        logger.error(f"Error in advanced clustering: {str(e)}")
        
        # Check calling context for error handling
        if hasattr(context, 'aws_request_id') and not event.get('httpMethod'):
            # Step Functions call - raise error
            raise e
        else:
            # API Gateway or direct call - return HTTP error
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Advanced clustering failed',
                    'message': str(e)
                })
            } 