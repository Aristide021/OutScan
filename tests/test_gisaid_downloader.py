"""
Unit tests for the GISAID downloader Lambda function
"""
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path to allow direct import of genomic_ingestion.gisaid_downloader
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from genomic_ingestion.gisaid_downloader import lambda_handler

class TestGisaidDownloader(unittest.TestCase):
    """Test suite for the GISAID downloader"""

    @patch('genomic_ingestion.gisaid_downloader.boto3.resource')
    @patch('genomic_ingestion.gisaid_downloader.boto3.client')
    @patch('genomic_ingestion.gisaid_downloader.requests.get')
    @patch('genomic_ingestion.gisaid_downloader.requests.post')
    def test_lambda_handler_success(self, mock_requests_post, mock_requests_get, mock_boto_client, mock_boto_resource):
        """Test successful invocation of the lambda_handler"""
        
        # Mock S3 client
        mock_s3 = MagicMock()
        mock_boto_client.return_value = mock_s3
        
        # Mock DynamoDB resource
        mock_dynamodb = MagicMock()
        mock_boto_resource.return_value = mock_dynamodb
        
        # Mock authentication response
        mock_auth_response = MagicMock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {'token': 'test-token'}
        mock_requests_post.return_value = mock_auth_response
        
        # Mock sequence fetch response (empty sequences)
        mock_sequences_response = MagicMock()
        mock_sequences_response.status_code = 200
        mock_sequences_response.json.return_value = {'sequences': []}
        mock_requests_get.return_value = mock_sequences_response
        
        # Mock environment variables
        os.environ['GENOMIC_DATA_BUCKET'] = 'test-bucket'
        os.environ['GISAID_USERNAME'] = 'test-user'
        os.environ['GISAID_PASSWORD'] = 'test-password'
        
        # Mock event and context
        mock_event = {}
        mock_context = MagicMock()
        
        # Call the handler
        response = lambda_handler(mock_event, mock_context)
        
        # Assert response (should be 200 with no sequences found message)
        self.assertEqual(response['statusCode'], 200)
        self.assertIn('No new sequences found', response['body'])

if __name__ == '__main__':
    unittest.main()
