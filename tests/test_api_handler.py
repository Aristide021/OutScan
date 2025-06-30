"""
Unit tests for the OutScan API handler
"""
import json
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path to allow direct import of api.api_handler
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.api_handler import lambda_handler

class TestApiHandler(unittest.TestCase):
    """Test suite for the main API handler"""

    def test_lambda_handler_success(self):
        """Test successful invocation of the lambda_handler"""
        
        # Mock event and context
        mock_event = {}
        mock_context = MagicMock()
        
        # Call the handler
        response = lambda_handler(mock_event, mock_context)
        
        # Assert status code
        self.assertEqual(response['statusCode'], 200)
        
        # Assert headers
        self.assertIn('Content-Type', response['headers'])
        self.assertEqual(response['headers']['Content-Type'], 'application/json')
        self.assertIn('Access-Control-Allow-Origin', response['headers'])
        self.assertEqual(response['headers']['Access-Control-Allow-Origin'], '*')
        
        # Parse body and assert content
        body = json.loads(response['body'])
        self.assertIsInstance(body, dict)
        
        # Check for key fields
        expected_keys = [
            "status",
            "timestamp",
            "sequences_analyzed",
            "variants_detected",
            "active_alerts",
            "processing_rate_per_minute",
            "daily_processing",
            "variant_prevalence",
            "system_info"
        ]
        for key in expected_keys:
            self.assertIn(key, body)
            
        # Check nested structures
        self.assertIsInstance(body['daily_processing'], list)
        self.assertIsInstance(body['variant_prevalence'], list)
        self.assertIsInstance(body['system_info'], dict)
        
        # Check that variant percentages sum to 100
        total_percentage = sum(item['percentage'] for item in body['variant_prevalence'])
        self.assertEqual(total_percentage, 100)

if __name__ == '__main__':
    unittest.main()
