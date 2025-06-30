"""
Lightweight SARS-CoV-2 Variant Clustering Engine
Hierarchical clustering implementation for variant detection using only NumPy/SciPy

This module implements clustering to identify emerging variant clusters from 
spike protein mutations using hierarchical clustering and distance matrices.
"""

import json
import boto3
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LightweightClusteringEngine:
    """Scipy-based clustering engine for variant detection"""
    
    def __init__(self):
        """Initialize the clustering engine with AWS services and parameters"""
        self.dynamodb = boto3.resource('dynamodb')
        self.variant_table = self.dynamodb.Table(os.environ.get('VARIANT_CLUSTERS_TABLE', 'VariantClusters'))
        self.mutation_table = self.dynamodb.Table(os.environ.get('MUTATION_LIBRARY_TABLE', 'MutationLibrary'))
        
        # Clustering parameters
        self.min_cluster_size = int(os.environ.get('MIN_CLUSTER_SIZE', '5'))
        self.distance_threshold = float(os.environ.get('CLUSTER_SELECTION_EPSILON', '0.7'))
        
        # Known variant signatures for comparison
        self.known_variants = {
            'Alpha': ['N501Y', 'D614G', 'P681H'],
            'Beta': ['N501Y', 'E484K', 'K417N', 'D614G'],
            'Gamma': ['N501Y', 'E484K', 'K417T', 'D614G'], 
            'Delta': ['L452R', 'T478K', 'D614G', 'P681R'],
            'Omicron': ['G142D', 'K417N', 'N440K', 'G446S', 'S477N', 'T478K', 
                       'E484A', 'Q493R', 'G496S', 'Q498R', 'N501Y', 'Y505H']
        }
        
        logger.info(f"Lightweight clustering engine initialized with min_cluster_size={self.min_cluster_size}")

    def load_variant_data(self, days_back: int = 30) -> List[Dict]:
        """Load variant data from DynamoDB for clustering analysis"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"Loading variant data from {start_date.date()} to {end_date.date()}")
            
            # Scan DynamoDB for recent variants
            response = self.variant_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('ProcessedTimestamp').between(
                    start_date.isoformat(), end_date.isoformat()
                )
            )
            
            variants = []
            for item in response['Items']:
                # Parse mutations
                mutations = item.get('Mutations', '[]')
                if isinstance(mutations, str):
                    try:
                        mutations = json.loads(mutations)
                    except:
                        mutations = []
                
                variants.append({
                    'sequence_id': item['SequenceID'],
                    'mutations': [m.get('notation', '') for m in mutations if m.get('notation')],
                    'collection_date': item.get('CollectionDate', ''),
                    'country': item.get('Country', 'Unknown'),
                    'lineage': item.get('Lineage', 'Unknown'),
                    'mutation_count': item.get('MutationCount', 0),
                    'critical_mutations': item.get('CriticalMutations', 0)
                })
            
            logger.info(f"Loaded {len(variants)} variant sequences")
            return variants
            
        except Exception as e:
            logger.error(f"Error loading from DynamoDB: {e}")
            return []

    def jaccard_distance(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard distance between two mutation sets"""
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return 1 - (intersection / union if union > 0 else 0)

    def calculate_distance_matrix(self, variants: List[Dict]) -> np.ndarray:
        """Calculate pairwise distance matrix for mutation signatures"""
        logger.info("Calculating pairwise distance matrix...")
        
        mutation_sets = [set(variant['mutations']) for variant in variants]
        n = len(mutation_sets)
        
        if n < 2:
            return np.array([[0]])
        
        # Create distance matrix using vectorized operations where possible
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dist = self.jaccard_distance(mutation_sets[i], mutation_sets[j])
                distances.append(dist)
        
        # Convert to square matrix
        distance_matrix = squareform(distances)
        
        logger.info(f"Distance matrix calculated: {n}x{n}")
        return distance_matrix

    def perform_hierarchical_clustering(self, variants: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Perform hierarchical clustering on variant data"""
        logger.info("Starting hierarchical clustering analysis...")
        
        if len(variants) < self.min_cluster_size:
            logger.warning(f"Insufficient data for clustering: {len(variants)} < {self.min_cluster_size}")
            return variants, {'total_clusters': 0, 'cluster_details': {}}
        
        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(variants)
        
        if distance_matrix.shape[0] < 2:
            return variants, {'total_clusters': 0, 'cluster_details': {}}
        
        # Perform hierarchical clustering using Ward linkage
        logger.info("Performing hierarchical clustering...")
        try:
            # Convert to condensed distance matrix for linkage
            condensed_distances = squareform(distance_matrix, checks=False)
            
            # Perform linkage
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Form flat clusters
            cluster_labels = fcluster(linkage_matrix, t=self.distance_threshold, criterion='distance')
            
            # Add cluster labels to variants
            for i, variant in enumerate(variants):
                variant['cluster_id'] = int(cluster_labels[i]) - 1  # 0-based indexing
                variant['cluster_probability'] = 1.0  # Hierarchical clustering gives hard assignments
            
            # Analyze clusters
            cluster_analysis = self.analyze_clusters(variants)
            
            logger.info(f"Clustering complete: {cluster_analysis['total_clusters']} clusters detected")
            return variants, cluster_analysis
            
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return variants, {'total_clusters': 0, 'cluster_details': {}}

    def analyze_clusters(self, variants: List[Dict]) -> Dict:
        """Analyze detected clusters for emergent variants"""
        logger.info("Analyzing clusters for emergent variants...")
        
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
            if cluster_id < 0 or len(cluster_variants) < self.min_cluster_size:
                continue
                
            # Find consensus mutations for this cluster
            all_mutations = []
            for variant in cluster_variants:
                all_mutations.extend(variant['mutations'])
            mutation_counts = Counter(all_mutations)
            
            # Mutations present in >50% of cluster members
            consensus_mutations = {mut for mut, count in mutation_counts.items() 
                                 if count > len(cluster_variants) * 0.5}
            
            # Compare to known variants
            similarities = self.compare_to_known_variants(consensus_mutations)
            max_similarity = max(similarities.values()) if similarities else 0
            
            # Calculate geographic and temporal distribution
            countries = Counter(v['country'] for v in cluster_variants)
            dates = [v['collection_date'] for v in cluster_variants if v['collection_date']]
            
            # Check if this is a potential emergent variant
            is_emergent = (
                max_similarity < 0.7 and  # Low similarity to known variants
                len(consensus_mutations) >= 3 and  # Has significant mutations
                len(cluster_variants) >= self.min_cluster_size and  # Sufficient cluster size
                len(countries) >= 2  # Geographic spread
            )
            
            cluster_info = {
                'size': len(cluster_variants),
                'consensus_mutations': list(consensus_mutations),
                'geographic_distribution': dict(countries),
                'temporal_span': {
                    'earliest': min(dates) if dates else '',
                    'latest': max(dates) if dates else '',
                    'span_days': (datetime.fromisoformat(max(dates)) - 
                                datetime.fromisoformat(min(dates))).days if len(dates) > 1 else 0
                },
                'known_variant_similarities': similarities,
                'max_known_similarity': max_similarity,
                'is_emergent': is_emergent,
                'risk_score': self.calculate_risk_score(cluster_variants, consensus_mutations),
                'growth_rate': self.estimate_growth_rate(cluster_variants)
            }
            
            analysis['cluster_details'][cluster_id] = cluster_info
            
            if is_emergent:
                analysis['emergent_variants'].append({
                    'cluster_id': cluster_id,
                    'variant_name': f"OutScan-{cluster_id}",
                    'consensus_mutations': list(consensus_mutations),
                    'size': len(cluster_variants),
                    'countries': list(countries.keys()),
                    'risk_score': cluster_info['risk_score']
                })
        
        logger.info(f"Analysis complete: {len(analysis['emergent_variants'])} emergent variants detected")
        return analysis

    def compare_to_known_variants(self, mutation_set: Set[str]) -> Dict[str, float]:
        """Compare mutation set to known variant signatures"""
        similarities = {}
        for variant_name, variant_mutations in self.known_variants.items():
            variant_set = set(variant_mutations)
            jaccard_sim = 1 - self.jaccard_distance(mutation_set, variant_set)
            similarities[variant_name] = jaccard_sim
        return similarities

    def calculate_risk_score(self, cluster_variants: List[Dict], consensus_mutations: Set[str]) -> float:
        """Calculate risk score for a cluster"""
        # Base score on cluster size
        size_score = min(len(cluster_variants) / 100, 1.0)
        
        # Score based on critical mutations
        critical_mutations = sum(v['critical_mutations'] for v in cluster_variants)
        critical_score = min(critical_mutations / (len(cluster_variants) * 5), 1.0)
        
        # Score based on geographic spread
        countries = set(v['country'] for v in cluster_variants)
        geo_score = min(len(countries) / 10, 1.0)
        
        # Score based on mutation novelty
        known_mutations = set()
        for variant_muts in self.known_variants.values():
            known_mutations.update(variant_muts)
        
        novel_mutations = consensus_mutations - known_mutations
        novelty_score = min(len(novel_mutations) / 5, 1.0)
        
        # Weighted combination
        risk_score = (0.3 * size_score + 0.3 * critical_score + 
                     0.2 * geo_score + 0.2 * novelty_score)
        
        return round(risk_score, 3)

    def estimate_growth_rate(self, cluster_variants: List[Dict]) -> float:
        """Estimate weekly growth rate of cluster"""
        dates = [v['collection_date'] for v in cluster_variants if v['collection_date']]
        
        if len(dates) < 2:
            return 0.0
        
        try:
            dates = sorted([datetime.fromisoformat(d) for d in dates])
            span_weeks = (dates[-1] - dates[0]).days / 7
            
            if span_weeks > 0:
                growth_rate = ((len(cluster_variants) - 1) / span_weeks) * 100
                return round(growth_rate, 1)
        except:
            pass
        
        return 0.0

    def save_clustering_results(self, variants: List[Dict], cluster_analysis: Dict) -> Dict:
        """Save clustering results to DynamoDB"""
        try:
            saved_clusters = 0
            
            # Update variant records with cluster information
            for variant in variants:
                if 'cluster_id' in variant:
                    try:
                        self.variant_table.update_item(
                            Key={'SequenceID': variant['sequence_id']},
                            UpdateExpression='SET ClusterID = :cid, ClusterProbability = :prob',
                            ExpressionAttributeValues={
                                ':cid': variant['cluster_id'],
                                ':prob': variant.get('cluster_probability', 1.0)
                            }
                        )
                        saved_clusters += 1
                    except Exception as e:
                        logger.error(f"Error updating variant {variant['sequence_id']}: {e}")
            
            logger.info(f"Saved clustering results for {saved_clusters} variants")
            
            return {
                'variants_processed': len(variants),
                'clusters_detected': cluster_analysis['total_clusters'],
                'emergent_variants': len(cluster_analysis['emergent_variants']),
                'records_updated': saved_clusters,
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error saving clustering results: {e}")
            return {'error': str(e)}

def lambda_handler(event, context):
    """
    Lambda handler for variant clustering analysis
    """
    logger.info("Starting variant clustering analysis")
    
    try:
        engine = LightweightClusteringEngine()
        
        # Load variant data
        days_back = event.get('days_back', 30)
        variants = engine.load_variant_data(days_back)
        
        if not variants:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'No variant data available for clustering',
                    'clusters_detected': 0
                })
            }
        
        # Perform clustering
        clustered_variants, cluster_analysis = engine.perform_hierarchical_clustering(variants)
        
        # Save results
        save_results = engine.save_clustering_results(clustered_variants, cluster_analysis)
        
        # Prepare response
        response = {
            'variants_analyzed': len(variants),
            'clusters_detected': cluster_analysis['total_clusters'],
            'emergent_variants': cluster_analysis['emergent_variants'],
            'cluster_details': cluster_analysis['cluster_details'],
            'save_results': save_results,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Clustering analysis complete: {cluster_analysis['total_clusters']} clusters")
        
        return {
            'statusCode': 200,
            'body': json.dumps(response, default=str)
        }
        
    except Exception as e:
        logger.error(f"Error in clustering analysis: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'Clustering analysis failed'
            })
        }

# Test function for local development
if __name__ == "__main__":
    test_event = {'days_back': 7}
    result = lambda_handler(test_event, None)
    print(f"Test result: {json.dumps(result, indent=2)}") 