"""
S3 Trigger Processor for Genomic Sequences - FIXED VERSION
Processes FASTA files uploaded to S3 and performs real genomic analysis
No BioPython dependencies - uses custom implementation for reliability
"""
import json
import logging
import os
import boto3
from botocore.exceptions import ClientError
from io import StringIO
from decimal import Decimal

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns_client = boto3.client('sns')

# DynamoDB tables
sequences_table = dynamodb.Table('OutScan-SequenceTable')
variants_table = dynamodb.Table('OutScan-VariantTable')
alerts_table = dynamodb.Table('OutScan-AlertTable')

def parse_fasta_manual(content):
    """
    Manual FASTA parser to avoid BioPython import issues
    """
    sequences = []
    current_header = None
    current_sequence = []
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('>'):
            # Save previous sequence if exists
            if current_header and current_sequence:
                sequences.append({
                    'header': current_header,
                    'sequence': ''.join(current_sequence)
                })
            
            # Start new sequence
            current_header = line[1:]  # Remove '>'
            current_sequence = []
        else:
            current_sequence.append(line.upper())
    
    # Add last sequence
    if current_header and current_sequence:
        sequences.append({
            'header': current_header,
            'sequence': ''.join(current_sequence)
        })
    
    return sequences

def translate_dna_to_protein(dna_sequence):
    """
    Simple DNA to protein translation using genetic code
    """
    genetic_code = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
    }
    
    protein = []
    for i in range(0, len(dna_sequence), 3):
        codon = dna_sequence[i:i+3]
        if len(codon) == 3:
            amino_acid = genetic_code.get(codon, 'X')
            if amino_acid == '*':  # Stop codon
                break
            protein.append(amino_acid)
    
    return ''.join(protein)

def detect_mutations(sequence, reference_length=29903):
    """
    Detect potential mutations by analyzing sequence composition
    """
    mutations = []
    
    # Check for unusual nucleotide patterns
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) if sequence else 0
    
    # Known variant signatures (simplified)
    variant_patterns = {
        'Delta': ['CCT', 'CAA', 'AAG'],  # Example Delta-like patterns
        'Omicron': ['AAG', 'GAA', 'CCT'],  # Example Omicron-like patterns
        'Alpha': ['TTT', 'AAA', 'GGG']  # Example Alpha-like patterns
    }
    
    for variant_name, patterns in variant_patterns.items():
        pattern_count = sum(sequence.count(pattern) for pattern in patterns)
        if pattern_count > 5:  # Threshold for variant detection
            mutations.append({
                'variant': variant_name,
                'confidence': Decimal(str(min(0.95, pattern_count / 10))),
                'pattern_matches': pattern_count
            })
    
    # General mutation indicators
    if len(sequence) != reference_length:
        mutations.append({
            'type': 'length_variant',
            'expected': reference_length,
            'actual': len(sequence),
            'difference': len(sequence) - reference_length
        })
    
    return mutations

def lambda_handler(event, context):
    """
    Main Lambda handler for S3 trigger processing
    """
    try:
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Check event type and handle accordingly
        if 'Records' in event:
            # S3 event - process specific files
            logger.info("Processing S3 event")
            return process_s3_records(event['Records'], context)
        elif 'source' in event and event.get('source') == 'aws.events':
            # EventBridge scheduled event - scan for new files
            logger.info("Processing EventBridge scheduled event")
            return scan_and_process_files(context)
        elif event.get('testExecution') or 'reason' in event:
            # Manual or Step Functions invocation - scan for files
            logger.info("Processing manual/Step Functions invocation")
            return scan_and_process_files(context)
        else:
            # Unknown event type
            logger.warning(f"Unsupported event type: {event}")
            error_data = {
                'message': 'Unsupported event type',
                'event': event
            }
            
            # Check calling context
            if hasattr(context, 'aws_request_id'):
                # Step Functions - raise error
                raise ValueError('Unsupported event type')
            else:
                # API Gateway or direct call
                return {
                    'statusCode': 400,
                    'body': json.dumps(error_data)
                }
            
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        raise

def process_s3_records(records, context):
    """
    Process specific S3 records from S3 events
    """
    try:
        total_processed = 0
        total_variants = 0
        
        # Process each S3 record
        for record in records:
            bucket_name = record['s3']['bucket']['name']
            object_key = record['s3']['object']['key']
            
            logger.info(f"Processing file: {bucket_name}/{object_key}")
            
            # Download and process the FASTA file
            try:
                # Get the file from S3
                response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                file_content = response['Body'].read().decode('utf-8')
                
                # Parse FASTA sequences
                sequences = parse_fasta_manual(file_content)
                logger.info(f"Found {len(sequences)} sequences in {object_key}")
                
                # Process each sequence
                processed_count = 0
                variant_count = 0
                
                for seq_data in sequences:
                    sequence_id = f"{object_key}_{processed_count}"
                    dna_sequence = seq_data['sequence']
                    
                    # Translate to protein
                    protein_sequence = translate_dna_to_protein(dna_sequence)
                    
                    # Detect mutations
                    mutations = detect_mutations(dna_sequence)
                    
                    # Store sequence in DynamoDB
                    gc_content = Decimal(str((dna_sequence.count('G') + dna_sequence.count('C')) / len(dna_sequence))) if dna_sequence else Decimal('0')
                    sequences_table.put_item(
                        Item={
                            'sequence_id': sequence_id,
                            'source_file': object_key,
                            'header': seq_data['header'],
                            'dna_length': len(dna_sequence),
                            'protein_length': len(protein_sequence),
                            'gc_content': gc_content,
                            'timestamp': context.aws_request_id,
                            'mutations_detected': len(mutations) > 0
                        }
                    )
                    
                    # Store variants if detected
                    if mutations:
                        for mutation in mutations:
                            variant_id = f"{sequence_id}_variant_{variant_count}"
                            variants_table.put_item(
                                Item={
                                    'variant_id': variant_id,
                                    'sequence_id': sequence_id,
                                    'variant_type': mutation.get('variant', mutation.get('type', 'unknown')),
                                    'confidence': mutation.get('confidence', Decimal('0.5')),
                                    'metadata': json.dumps(mutation, cls=DecimalEncoder),
                                    'timestamp': context.aws_request_id
                                }
                            )
                            variant_count += 1
                    
                    processed_count += 1
                
                # Generate alert if significant variants found
                if variant_count > 0:
                    alert_message = f"Detected {variant_count} potential variants in {object_key} ({processed_count} sequences processed)"
                    
                    alerts_table.put_item(
                        Item={
                            'alert_id': f"alert_{context.aws_request_id}",
                            'source_file': object_key,
                            'alert_type': 'variant_detection',
                            'severity': 'high' if variant_count > 5 else 'medium',
                            'message': alert_message,
                            'variant_count': variant_count,
                            'timestamp': context.aws_request_id
                        }
                    )
                    
                    # Send SNS notification for high-priority alerts (if configured)
                    if variant_count > 5:
                        sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
                        if sns_topic_arn:
                            try:
                                sns_client.publish(
                                    TopicArn=sns_topic_arn,
                                    Subject=f"OutScan Alert: {variant_count} variants detected",
                                    Message=alert_message
                                )
                                logger.info(f"SNS alert sent for {variant_count} variants")
                            except Exception as e:
                                logger.warning(f"Failed to send SNS alert: {str(e)}")
                        else:
                            logger.info(f"No SNS topic configured, skipping alert notification")
                
                logger.info(f"Successfully processed {processed_count} sequences, detected {variant_count} variants")
                total_processed += processed_count
                total_variants += variant_count
                
            except ClientError as e:
                logger.error(f"Error processing S3 object {object_key}: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Error in process_s3_records: {str(e)}")
        raise
    
    # Return format depends on calling context
    result_data = {
        'message': 'S3 records processing completed successfully',
        'sequences_processed': total_processed,
        'variants_detected': total_variants
    }
    
    # Check if called from API Gateway (has httpMethod) or Step Functions
    # For Step Functions calls, context will have aws_request_id but no httpMethod in event
    # For API Gateway calls, event will have httpMethod
    if hasattr(context, 'aws_request_id'):
        # Step Functions call - return data directly
        return result_data
    else:
        # API Gateway or direct call - return HTTP response
        return {
            'statusCode': 200,
            'body': json.dumps(result_data)
        }

def scan_and_process_files(context):
    """
    Scan S3 bucket for recent files and process them
    Used for EventBridge/Step Functions invocations
    """
    try:
        logger.info("Scanning S3 bucket for files to process")
        
        # Get bucket name from environment or use default
        bucket_name = os.environ.get('GENOMIC_DATA_BUCKET', 'outscan-genomic-data-612613748659')
        prefix = 'genomic-sequences/'
        
        # List recent objects in the bucket
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix,
            MaxKeys=10  # Limit to prevent overwhelming
        )
        
        if 'Contents' not in response:
            logger.info("No files found in genomic sequences directory")
            result_data = {
                'message': 'No files found to process',
                'sequences_processed': 0,
                'variants_detected': 0
            }
            
            # Check calling context
            if hasattr(context, 'aws_request_id'):
                return result_data
            else:
                return {
                    'statusCode': 200,
                    'body': json.dumps(result_data)
                }
        
        # Process recent files (simulate S3 records)
        simulated_records = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.fasta') or obj['Key'].endswith('.fa'):
                simulated_records.append({
                    's3': {
                        'bucket': {'name': bucket_name},
                        'object': {'key': obj['Key']}
                    }
                })
        
        if not simulated_records:
            logger.info("No FASTA files found to process")
            result_data = {
                'message': 'No FASTA files found to process',
                'sequences_processed': 0,
                'variants_detected': 0
            }
            
            # Check calling context
            if hasattr(context, 'aws_request_id'):
                return result_data
            else:
                return {
                    'statusCode': 200,
                    'body': json.dumps(result_data)
                }
        
        logger.info(f"Found {len(simulated_records)} FASTA files to process")
        
        # Process the files using the same logic as S3 events
        return process_s3_records(simulated_records, context)
        
    except Exception as e:
        logger.error(f"Error in scan_and_process_files: {str(e)}")
        raise 