"""
GISAID Genomic Data Downloader
Polls GISAID/NCBI APIs for real-time sequence data
"""
import json
import boto3
import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GISAIDDownloader:
    """
    Handles downloading and processing of genomic sequences from GISAID API
    """
    
    def __init__(self, 
                 gisaid_username: str,
                 gisaid_password: str,
                 s3_bucket: str,
                 min_sequence_length: int = 29000):
        self.gisaid_username = gisaid_username
        self.gisaid_password = gisaid_password
        self.s3_bucket = s3_bucket
        self.min_sequence_length = min_sequence_length
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        
        # API endpoints
        self.gisaid_api_base = "https://www.epicov.org/epi3/frontend"
        self.ncbi_api_base = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha"
        
    def authenticate_gisaid(self) -> Optional[str]:
        """Authenticate with GISAID and return session token"""
        try:
            auth_payload = {
                'username': self.gisaid_username,
                'password': self.gisaid_password
            }
            
            response = requests.post(
                f"{self.gisaid_api_base}/authentication",
                json=auth_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                token = response.json().get('token')
                logger.info("Successfully authenticated with GISAID")
                return token
            else:
                logger.error(f"GISAID authentication failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error authenticating with GISAID: {str(e)}")
            return None
    
    def fetch_recent_sequences(self, 
                              hours_back: int = 24,
                              lineage_filter: Optional[str] = None) -> List[Dict]:
        """
        Fetch sequences uploaded in the last N hours
        """
        sequences = []
        token = self.authenticate_gisaid()
        
        if not token:
            return sequences
            
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=hours_back)
            
            headers = {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            }
            
            query_params = {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'complete': True,
                'high_coverage': True
            }
            
            if lineage_filter:
                query_params['lineage'] = lineage_filter
            
            response = requests.get(
                f"{self.gisaid_api_base}/sequences",
                headers=headers,
                params=query_params,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                sequences = data.get('sequences', [])
                logger.info(f"Retrieved {len(sequences)} sequences from GISAID")
            else:
                logger.error(f"Failed to fetch sequences: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching sequences: {str(e)}")
            
        return sequences
    
    def validate_sequence(self, sequence: Dict) -> bool:
        """
        Validate genomic sequence quality
        """
        try:
            seq_data = sequence.get('sequence', '')
            metadata = sequence.get('metadata', {})
            
            # Check sequence length
            if len(seq_data) < self.min_sequence_length:
                logger.warning(f"Sequence {metadata.get('accession')} too short: {len(seq_data)} bp")
                return False
            
            # Check for excessive N's (ambiguous nucleotides)
            n_count = seq_data.upper().count('N')
            n_percentage = (n_count / len(seq_data)) * 100
            
            if n_percentage > 5.0:  # More than 5% N's
                logger.warning(f"Sequence {metadata.get('accession')} has {n_percentage:.1f}% N's")
                return False
            
            # Check required metadata fields
            required_fields = ['collection_date', 'country', 'lineage']
            for field in required_fields:
                if not metadata.get(field):
                    logger.warning(f"Sequence {metadata.get('accession')} missing {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating sequence: {str(e)}")
            return False
    
    def upload_to_s3(self, sequences: List[Dict]) -> List[str]:
        """
        Upload validated sequences to S3 for processing
        """
        uploaded_keys = []
        
        for sequence in sequences:
            try:
                metadata = sequence.get('metadata', {})
                accession = metadata.get('accession', 'unknown')
                
                # Create S3 key with date partitioning
                collection_date = metadata.get('collection_date', datetime.now().strftime('%Y-%m-%d'))
                s3_key = f"genomic-sequences/{collection_date}/{accession}.fasta"
                
                # Format as FASTA
                fasta_content = f">{accession}\n{sequence['sequence']}\n"
                
                # Add metadata as object tags
                tags = {
                    'lineage': metadata.get('lineage', ''),
                    'country': metadata.get('country', ''),
                    'collection_date': collection_date,
                    'upload_timestamp': datetime.now().isoformat()
                }
                
                tag_set = [{'Key': k, 'Value': str(v)} for k, v in tags.items()]
                
                # Upload to S3
                self.s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=fasta_content.encode('utf-8'),
                    ContentType='text/plain',
                    Tagging='&'.join([f"{k}={v}" for k, v in tags.items()])
                )
                
                uploaded_keys.append(s3_key)
                logger.info(f"Uploaded sequence {accession} to S3: {s3_key}")
                
            except ClientError as e:
                logger.error(f"S3 upload failed for {accession}: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error uploading {accession}: {str(e)}")
        
        return uploaded_keys

def lambda_handler(event, context):
    """
    AWS Lambda handler for scheduled genomic data downloads
    """
    try:
        # Get configuration from environment variables
        import os
        
        gisaid_username = os.environ.get('GISAID_USERNAME')
        gisaid_password = os.environ.get('GISAID_PASSWORD')
        s3_bucket = os.environ.get('GENOMIC_DATA_BUCKET')
        
        if not all([gisaid_username, gisaid_password, s3_bucket]):
            raise ValueError("Missing required environment variables")
        
        # Initialize downloader
        downloader = GISAIDDownloader(
            gisaid_username=gisaid_username,
            gisaid_password=gisaid_password,
            s3_bucket=s3_bucket
        )
        
        # Fetch sequences from last 24 hours
        sequences = downloader.fetch_recent_sequences(hours_back=24)
        
        if not sequences:
            logger.info("No new sequences found")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'No new sequences found', 'count': 0})
            }
        
        # Validate and filter sequences
        valid_sequences = [seq for seq in sequences if downloader.validate_sequence(seq)]
        
        logger.info(f"Validated {len(valid_sequences)} out of {len(sequences)} sequences")
        
        # Upload to S3
        uploaded_keys = downloader.upload_to_s3(valid_sequences)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed genomic data',
                'total_sequences': len(sequences),
                'valid_sequences': len(valid_sequences),
                'uploaded_sequences': len(uploaded_keys),
                's3_keys': uploaded_keys
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
    import os
    
    # Mock event and context for testing
    test_event = {}
    test_context = type('Context', (), {'aws_request_id': 'test-request-id'})()
    
    # Set test environment variables
    os.environ['GISAID_USERNAME'] = 'test_user'
    os.environ['GISAID_PASSWORD'] = 'test_password'
    os.environ['GENOMIC_DATA_BUCKET'] = 'test-genomic-bucket'
    
    result = lambda_handler(test_event, test_context)
    print(json.dumps(result, indent=2)) 