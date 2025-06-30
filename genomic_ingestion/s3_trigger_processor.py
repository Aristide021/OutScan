"""
S3 Trigger Processor for Genomic Sequences
Processes FASTA/FASTQ files uploaded to S3 and extracts spike proteins
"""
import json
import boto3
import logging
import re
from io import StringIO
from typing import List, Dict, Tuple, Optional
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip
import urllib.parse

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GenomicProcessor:
    """
    Processes genomic sequences and extracts spike protein mutations
    """
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        
        # Reference genome (Wuhan-1)
        self.reference_sequence = self._load_reference_genome()
        self.spike_start = 21563  # Spike protein start position
        self.spike_end = 25384    # Spike protein end position
        
        # DynamoDB tables
        self.variant_table = self.dynamodb.Table('VariantClusters')
        self.mutation_table = self.dynamodb.Table('MutationLibrary')
    
    def _load_reference_genome(self) -> str:
        """Load SARS-CoV-2 reference genome (EPI_ISL_402124)"""
        # In production, this would be loaded from S3 or embedded
        # For now, returning spike region coordinates
        return "REFERENCE_GENOME_PLACEHOLDER"
    
    def parse_fasta_from_s3(self, bucket: str, key: str) -> List[SeqRecord]:
        """
        Parse FASTA/FASTQ file from S3
        """
        sequences = []
        
        try:
            # Download file from S3
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read()
            
            # Handle gzip compression
            if key.endswith('.gz'):
                content = gzip.decompress(content)
            
            # Decode content
            content = content.decode('utf-8')
            
            # Determine file format
            file_format = 'fasta'
            if key.endswith(('.fastq', '.fq')):
                file_format = 'fastq'
            
            # Parse sequences
            sequences = list(SeqIO.parse(StringIO(content), file_format))
            
            logger.info(f"Parsed {len(sequences)} sequences from {bucket}/{key}")
            
        except Exception as e:
            logger.error(f"Error parsing file {bucket}/{key}: {str(e)}")
            
        return sequences
    
    def extract_spike_protein(self, genome_sequence: str) -> Optional[str]:
        """
        Extract spike protein coding sequence from full genome
        """
        try:
            # Extract spike region (positions 21563-25384)
            if len(genome_sequence) >= self.spike_end:
                spike_sequence = genome_sequence[self.spike_start-1:self.spike_end]
                
                # Translate to amino acids
                dna_seq = Seq(spike_sequence)
                protein_seq = str(dna_seq.translate(to_stop=True))
                
                # Validate spike protein length (should be ~1273 amino acids)
                if 1200 <= len(protein_seq) <= 1300:
                    return protein_seq
                else:
                    logger.warning(f"Spike protein length unusual: {len(protein_seq)} aa")
                    return protein_seq
            else:
                logger.warning(f"Genome sequence too short: {len(genome_sequence)} bp")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting spike protein: {str(e)}")
            return None
    
    def identify_mutations(self, spike_sequence: str) -> List[Dict]:
        """
        Identify mutations in spike protein compared to reference
        """
        mutations = []
        
        try:
            # Reference spike protein sequence (Wuhan-1)
            reference_spike = self._get_reference_spike()
            
            if not reference_spike:
                logger.error("Reference spike sequence not available")
                return mutations
            
            # Compare sequences position by position
            min_length = min(len(spike_sequence), len(reference_spike))
            
            for position in range(min_length):
                ref_aa = reference_spike[position]
                var_aa = spike_sequence[position]
                
                if ref_aa != var_aa and var_aa != 'X':  # X = unknown amino acid
                    mutation = {
                        'position': position + 1,  # 1-based position
                        'reference': ref_aa,
                        'variant': var_aa,
                        'notation': f"{ref_aa}{position + 1}{var_aa}",
                        'is_critical': self._is_critical_mutation(position + 1, var_aa)
                    }
                    mutations.append(mutation)
            
            # Check for insertions/deletions
            if len(spike_sequence) != len(reference_spike):
                logger.info(f"Length difference detected: {len(spike_sequence)} vs {len(reference_spike)}")
            
            logger.info(f"Identified {len(mutations)} mutations in spike protein")
            
        except Exception as e:
            logger.error(f"Error identifying mutations: {str(e)}")
            
        return mutations
    
    def _get_reference_spike(self) -> str:
        """Get reference spike protein sequence"""
        # In production, this would be loaded from DynamoDB or S3
        # Returning a placeholder - should be the actual Wuhan-1 spike sequence
        return "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
    
    def _is_critical_mutation(self, position: int, amino_acid: str) -> bool:
        """
        Check if mutation is in WHO-critical positions
        """
        # WHO-critical mutations for immune escape and virulence
        critical_positions = {
            69, 70, 95, 142, 143, 144, 145, 211, 214, 215, 242, 243, 244, 245,
            417, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450,
            452, 453, 454, 455, 456, 460, 477, 478, 484, 485, 486, 487, 489,
            490, 491, 493, 494, 495, 496, 498, 499, 500, 501, 502, 505, 614,
            655, 681, 701, 716, 718, 796, 950, 982, 1027
        }
        
        return position in critical_positions
    
    def store_variant_data(self, sequence_id: str, mutations: List[Dict], metadata: Dict):
        """
        Store variant information in DynamoDB
        """
        try:
            # Create mutation signature
            mutation_signature = ",".join([m['notation'] for m in mutations])
            
            # Store in VariantClusters table
            variant_item = {
                'SequenceID': sequence_id,
                'MutationSignature': mutation_signature,
                'MutationCount': len(mutations),
                'CriticalMutations': len([m for m in mutations if m['is_critical']]),
                'Country': metadata.get('country', 'Unknown'),
                'CollectionDate': metadata.get('collection_date', ''),
                'Lineage': metadata.get('lineage', 'Unknown'),
                'ProcessedTimestamp': json.dumps(metadata.get('processed_timestamp', '')),
                'Mutations': json.dumps(mutations)
            }
            
            self.variant_table.put_item(Item=variant_item)
            
            # Update mutation library
            for mutation in mutations:
                self._update_mutation_frequency(mutation)
            
            logger.info(f"Stored variant data for {sequence_id}")
            
        except Exception as e:
            logger.error(f"Error storing variant data: {str(e)}")
    
    def _update_mutation_frequency(self, mutation: Dict):
        """
        Update mutation frequency in mutation library
        """
        try:
            mutation_key = mutation['notation']
            
            # Update or create mutation record
            response = self.mutation_table.update_item(
                Key={'MutationID': mutation_key},
                UpdateExpression='ADD Frequency :inc SET AA_Position = :pos, ImpactScore = :score, DrugResistance = :drug',
                ExpressionAttributeValues={
                    ':inc': 1,
                    ':pos': mutation['position'],
                    ':score': 0.5 if mutation['is_critical'] else 0.1,  # Placeholder scoring
                    ':drug': 'Unknown'  # Would be populated from research data
                },
                ReturnValues='ALL_NEW'
            )
            
        except Exception as e:
            logger.error(f"Error updating mutation frequency: {str(e)}")

def lambda_handler(event, context):
    """
    AWS Lambda handler for S3 trigger events
    """
    try:
        processor = GenomicProcessor()
        processed_files = []
        
        # Process each S3 event record
        for record in event.get('Records', []):
            if record.get('eventSource') == 'aws:s3':
                bucket = record['s3']['bucket']['name']
                key = urllib.parse.unquote_plus(record['s3']['object']['key'])
                
                logger.info(f"Processing file: {bucket}/{key}")
                
                # Parse sequences from S3 file
                sequences = processor.parse_fasta_from_s3(bucket, key)
                
                for seq_record in sequences:
                    try:
                        # Extract spike protein
                        spike_protein = processor.extract_spike_protein(str(seq_record.seq))
                        
                        if spike_protein:
                            # Identify mutations
                            mutations = processor.identify_mutations(spike_protein)
                            
                            # Extract metadata from sequence description
                            metadata = {
                                'sequence_id': seq_record.id,
                                'description': seq_record.description,
                                'country': 'Unknown',  # Would parse from description
                                'collection_date': '',  # Would parse from description
                                'lineage': 'Unknown',   # Would parse from description
                                'processed_timestamp': context.aws_request_id
                            }
                            
                            # Store variant data
                            processor.store_variant_data(
                                seq_record.id, 
                                mutations, 
                                metadata
                            )
                            
                            processed_files.append({
                                'sequence_id': seq_record.id,
                                'mutation_count': len(mutations),
                                'critical_mutations': len([m for m in mutations if m['is_critical']])
                            })
                            
                    except Exception as e:
                        logger.error(f"Error processing sequence {seq_record.id}: {str(e)}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully processed genomic sequences',
                'processed_sequences': len(processed_files),
                'details': processed_files
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
    test_event = {
        'Records': [{
            'eventSource': 'aws:s3',
            's3': {
                'bucket': {'name': 'test-genomic-bucket'},
                'object': {'key': 'genomic-sequences/2025-06-15/test_sequence.fasta'}
            }
        }]
    }
    
    test_context = type('Context', (), {'aws_request_id': 'test-request-id'})()
    
    result = lambda_handler(test_event, test_context)
    print(json.dumps(result, indent=2)) 