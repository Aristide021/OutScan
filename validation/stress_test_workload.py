"""
Stress Test Workload Generator
Tests OutScan system performance with high-volume genomic data
"""
import asyncio
import aiohttp
import json
import time
import random
import string
from typing import List, Dict
import boto3
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StressTestGenerator:
    """
    Generate high-volume workloads to test OutScan scalability
    """
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.lambda_client = boto3.client('lambda')
        self.stepfunctions_client = boto3.client('stepfunctions')
        
        # Test configuration
        self.genomic_bucket = 'outscan-genomic-data'
        self.test_results = []
        
    def generate_synthetic_sequence(self, length: int = 29903) -> str:
        """
        Generate synthetic SARS-CoV-2 genome sequence with mutations
        """
        # Base SARS-CoV-2 reference (simplified)
        nucleotides = ['A', 'T', 'G', 'C']
        sequence = ''.join(random.choices(nucleotides, k=length))
        
        # Add some realistic patterns
        # Add ORF1ab region
        sequence = sequence[:21563] + self._generate_spike_region() + sequence[25384:]
        
        return sequence
    
    def _generate_spike_region(self) -> str:
        """
        Generate spike protein region with potential mutations
        """
        # Spike region is ~3821 nucleotides
        spike_length = 3821
        nucleotides = ['A', 'T', 'G', 'C']
        
        # Generate base sequence
        spike_seq = ''.join(random.choices(nucleotides, k=spike_length))
        
        # Introduce mutations at known positions (simplified)
        mutation_positions = [1501, 1452, 1433, 1250, 614]  # Simplified positions
        
        for pos in mutation_positions:
            if pos < len(spike_seq) and random.random() < 0.3:  # 30% chance of mutation
                spike_seq = spike_seq[:pos] + random.choice(nucleotides) + spike_seq[pos+1:]
        
        return spike_seq
    
    def create_test_fasta(self, sequence_id: str, sequence: str, metadata: Dict) -> str:
        """
        Create FASTA format string with metadata
        """
        header = f">{sequence_id}|{metadata.get('country', 'Unknown')}|{metadata.get('collection_date', '2025-06-01')}|{metadata.get('lineage', 'Unknown')}"
        return f"{header}\n{sequence}\n"
    
    async def upload_sequences_batch(self, session: aiohttp.ClientSession, 
                                   batch_size: int, batch_id: int) -> Dict:
        """
        Upload a batch of synthetic sequences to S3
        """
        start_time = time.time()
        upload_results = []
        
        try:
            # Generate batch of sequences
            sequences = []
            for i in range(batch_size):
                sequence_id = f"STRESS_TEST_{batch_id}_{i:06d}"
                sequence = self.generate_synthetic_sequence()
                
                metadata = {
                    'country': random.choice(['United States', 'United Kingdom', 'Germany', 'Canada', 'Australia']),
                    'collection_date': f"2025-06-{random.randint(1, 28):02d}",
                    'lineage': random.choice(['B.1.1.7', 'B.1.617.2', 'B.1.1.529', 'XBB.1.5'])
                }
                
                fasta_content = self.create_test_fasta(sequence_id, sequence, metadata)
                sequences.append((sequence_id, fasta_content))
            
            # Upload to S3 in parallel
            upload_tasks = []
            for sequence_id, fasta_content in sequences:
                s3_key = f"stress-test/{time.strftime('%Y-%m-%d')}/{sequence_id}.fasta"
                
                # Use boto3 in thread pool for S3 uploads
                task = asyncio.get_event_loop().run_in_executor(
                    None,
                    self._upload_to_s3,
                    s3_key,
                    fasta_content
                )
                upload_tasks.append(task)
            
            # Wait for all uploads to complete
            upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)
            
            successful_uploads = sum(1 for result in upload_results if result is True)
            
            end_time = time.time()
            
            return {
                'batch_id': batch_id,
                'batch_size': batch_size,
                'successful_uploads': successful_uploads,
                'failed_uploads': batch_size - successful_uploads,
                'duration_seconds': end_time - start_time,
                'throughput_per_second': successful_uploads / (end_time - start_time) if (end_time - start_time) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Batch {batch_id} failed: {str(e)}")
            return {
                'batch_id': batch_id,
                'batch_size': batch_size,
                'error': str(e),
                'successful_uploads': 0,
                'failed_uploads': batch_size
            }
    
    def _upload_to_s3(self, s3_key: str, content: str) -> bool:
        """
        Upload single file to S3 (for thread pool execution)
        """
        try:
            self.s3_client.put_object(
                Bucket=self.genomic_bucket,
                Key=s3_key,
                Body=content.encode('utf-8'),
                ContentType='text/plain',
                Metadata={
                    'test_type': 'stress_test',
                    'upload_time': str(time.time())
                }
            )
            return True
        except Exception as e:
            logger.error(f"S3 upload failed for {s3_key}: {str(e)}")
            return False
    
    async def run_stress_test(self, target_sequences: int = 100000, 
                            batch_size: int = 1000, 
                            max_concurrent_batches: int = 50) -> Dict:
        """
        Run comprehensive stress test
        """
        logger.info(f"Starting stress test: {target_sequences} sequences, {batch_size} per batch")
        
        start_time = time.time()
        test_results = {
            'test_configuration': {
                'target_sequences': target_sequences,
                'batch_size': batch_size,
                'max_concurrent_batches': max_concurrent_batches,
                'start_time': start_time
            },
            'batch_results': [],
            'performance_metrics': {}
        }
        
        # Calculate number of batches needed
        num_batches = (target_sequences + batch_size - 1) // batch_size
        
        # Create semaphore to limit concurrent batches
        semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        async def run_batch_with_semaphore(session, batch_id):
            async with semaphore:
                return await self.upload_sequences_batch(session, batch_size, batch_id)
        
        # Run batches
        async with aiohttp.ClientSession() as session:
            batch_tasks = [
                run_batch_with_semaphore(session, batch_id) 
                for batch_id in range(num_batches)
            ]
            
            # Execute all batches
            batch_results = await asyncio.gather(*batch_tasks)
            test_results['batch_results'] = batch_results
        
        # Calculate overall metrics
        end_time = time.time()
        total_duration = end_time - start_time
        
        successful_sequences = sum(result.get('successful_uploads', 0) for result in batch_results)
        failed_sequences = sum(result.get('failed_uploads', 0) for result in batch_results)
        
        test_results['performance_metrics'] = {
            'total_duration_seconds': total_duration,
            'total_sequences_uploaded': successful_sequences,
            'total_sequences_failed': failed_sequences,
            'overall_throughput_per_second': successful_sequences / total_duration if total_duration > 0 else 0,
            'success_rate': successful_sequences / target_sequences if target_sequences > 0 else 0,
            'average_batch_duration': sum(result.get('duration_seconds', 0) for result in batch_results) / len(batch_results) if batch_results else 0
        }
        
        logger.info(f"Stress test completed: {successful_sequences}/{target_sequences} sequences uploaded in {total_duration:.1f}s")
        
        return test_results
    
    def test_lambda_scalability(self, concurrent_invocations: int = 100) -> Dict:
        """
        Test Lambda function scalability under load
        """
        logger.info(f"Testing Lambda scalability with {concurrent_invocations} concurrent invocations")
        
        start_time = time.time()
        
        # Prepare test payloads
        test_payloads = []
        for i in range(concurrent_invocations):
            payload = {
                'test_id': f'lambda_test_{i}',
                'mutations': ['N501Y', 'E484K', 'L452R'],
                'cluster_growth': random.uniform(10.0, 50.0),
                'geographic_spread': random.randint(1, 8)
            }
            test_payloads.append(payload)
        
        # Invoke Lambda functions concurrently
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for payload in test_payloads:
                future = executor.submit(self._invoke_lambda, 'OutScan-BedrockInferenceFunction', payload)
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e)})
        
        end_time = time.time()
        
        # Analyze results
        successful_invocations = sum(1 for result in results if 'error' not in result)
        failed_invocations = concurrent_invocations - successful_invocations
        
        return {
            'test_type': 'lambda_scalability',
            'concurrent_invocations': concurrent_invocations,
            'successful_invocations': successful_invocations,
            'failed_invocations': failed_invocations,
            'success_rate': successful_invocations / concurrent_invocations,
            'total_duration': end_time - start_time,
            'average_response_time': (end_time - start_time) / concurrent_invocations
        }
    
    def _invoke_lambda(self, function_name: str, payload: Dict) -> Dict:
        """
        Invoke Lambda function with payload
        """
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read())
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def test_step_functions_throughput(self, concurrent_executions: int = 20) -> Dict:
        """
        Test Step Functions workflow throughput
        """
        logger.info(f"Testing Step Functions with {concurrent_executions} concurrent executions")
        
        start_time = time.time()
        execution_arns = []
        
        # Start concurrent executions
        for i in range(concurrent_executions):
            try:
                response = self.stepfunctions_client.start_execution(
                    stateMachineArn='arn:aws:states:us-east-1:123456789012:stateMachine:VariantAnalysisWorkflow',
                    name=f'stress_test_execution_{i}_{int(time.time())}',
                    input=json.dumps({
                        'test_execution': True,
                        'mutations': ['N501Y', 'E484K'],
                        'sequence_count': 100
                    })
                )
                execution_arns.append(response['executionArn'])
            except Exception as e:
                logger.error(f"Failed to start execution {i}: {str(e)}")
        
        # Wait for executions to complete
        completed_executions = []
        for arn in execution_arns:
            try:
                # Poll execution status
                max_wait = 300  # 5 minutes
                wait_time = 0
                
                while wait_time < max_wait:
                    response = self.stepfunctions_client.describe_execution(executionArn=arn)
                    status = response['status']
                    
                    if status in ['SUCCEEDED', 'FAILED', 'TIMED_OUT', 'ABORTED']:
                        completed_executions.append({
                            'arn': arn,
                            'status': status,
                            'duration': (response.get('stopDate', response['startDate']) - response['startDate']).total_seconds()
                        })
                        break
                    
                    time.sleep(5)
                    wait_time += 5
                    
            except Exception as e:
                logger.error(f"Error monitoring execution {arn}: {str(e)}")
        
        end_time = time.time()
        
        successful_executions = sum(1 for exec in completed_executions if exec['status'] == 'SUCCEEDED')
        
        return {
            'test_type': 'step_functions_throughput',
            'concurrent_executions': concurrent_executions,
            'successful_executions': successful_executions,
            'completed_executions': len(completed_executions),
            'success_rate': successful_executions / concurrent_executions,
            'total_test_duration': end_time - start_time,
            'execution_details': completed_executions
        }
    
    def generate_performance_report(self, test_results: List[Dict]) -> str:
        """
        Generate comprehensive performance report
        """
        report = f"""
# OutScan Stress Test Performance Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}

## Test Summary
"""
        
        for test in test_results:
            test_type = test.get('test_type', 'Unknown')
            
            if test_type == 'genomic_upload':
                metrics = test['performance_metrics']
                report += f"""
### Genomic Data Upload Test
- **Target Sequences**: {test['test_configuration']['target_sequences']:,}
- **Successful Uploads**: {metrics['total_sequences_uploaded']:,}
- **Success Rate**: {metrics['success_rate']:.1%}
- **Total Duration**: {metrics['total_duration_seconds']:.1f} seconds
- **Throughput**: {metrics['overall_throughput_per_second']:.1f} sequences/second
- **Average Batch Duration**: {metrics['average_batch_duration']:.2f} seconds
"""
            
            elif test_type == 'lambda_scalability':
                report += f"""
### Lambda Scalability Test
- **Concurrent Invocations**: {test['concurrent_invocations']}
- **Successful Invocations**: {test['successful_invocations']}
- **Success Rate**: {test['success_rate']:.1%}
- **Average Response Time**: {test['average_response_time']:.2f} seconds
"""
            
            elif test_type == 'step_functions_throughput':
                report += f"""
### Step Functions Throughput Test
- **Concurrent Executions**: {test['concurrent_executions']}
- **Successful Executions**: {test['successful_executions']}
- **Success Rate**: {test['success_rate']:.1%}
- **Total Test Duration**: {test['total_test_duration']:.1f} seconds
"""
        
        return report

async def main():
    """
    Run comprehensive stress test suite
    """
    generator = StressTestGenerator()
    all_results = []
    
    # Test 1: High-volume genomic data upload
    print("=== RUNNING GENOMIC UPLOAD STRESS TEST ===")
    upload_results = await generator.run_stress_test(
        target_sequences=10000,  # Start with 10K for demo
        batch_size=500,
        max_concurrent_batches=20
    )
    upload_results['test_type'] = 'genomic_upload'
    all_results.append(upload_results)
    
    # Test 2: Lambda function scalability
    print("\n=== RUNNING LAMBDA SCALABILITY TEST ===")
    lambda_results = generator.test_lambda_scalability(concurrent_invocations=50)
    all_results.append(lambda_results)
    
    # Test 3: Step Functions throughput
    print("\n=== RUNNING STEP FUNCTIONS THROUGHPUT TEST ===")
    sfn_results = generator.test_step_functions_throughput(concurrent_executions=10)
    all_results.append(sfn_results)
    
    # Generate report
    report = generator.generate_performance_report(all_results)
    print("\n" + report)
    
    # Save results to S3
    s3_client = boto3.client('s3')
    s3_key = f"stress-test-results/{time.strftime('%Y-%m-%d')}/stress_test_results.json"
    
    s3_client.put_object(
        Bucket='outscan-analysis-results',
        Key=s3_key,
        Body=json.dumps(all_results, indent=2, default=str),
        ContentType='application/json'
    )
    
    print(f"\nResults saved to: s3://outscan-analysis-results/{s3_key}")

if __name__ == "__main__":
    asyncio.run(main()) 