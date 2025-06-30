"""
Amazon Bedrock Inference for Mutation Impact Prediction
Uses Claude 3 to predict functional impact of spike protein mutations
"""
import json
import boto3
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BedrockMutationAnalyzer:
    """
    Analyzes mutation impact using Amazon Bedrock's Claude 3
    """
    
    def __init__(self, model_id: str = "anthropic.claude-3-sonnet-20250229-v1:0"):
        self.bedrock_client = boto3.client('bedrock-runtime')
        self.model_id = model_id
        self.dynamodb = boto3.resource('dynamodb')
        self.mutation_table = self.dynamodb.Table('MutationLibrary')
        
        # Load known mutation impacts for validation
        self.known_mutations = self._load_known_mutations()
    
    def _load_known_mutations(self) -> Dict[str, Dict]:
        """
        Load database of known mutation impacts for training/validation
        """
        # In production, this would load from DynamoDB or S3
        # For now, using key mutations from literature
        return {
            'N501Y': {
                'immune_escape': 0.8,
                'transmissibility': 1.5,
                'virulence': 0.2,
                'drug_resistance': 0.1,
                'receptor_binding': 1.3,
                'description': 'Alpha/Beta/Gamma variant - increased ACE2 binding'
            },
            'E484K': {
                'immune_escape': 0.9,
                'transmissibility': 0.3,
                'virulence': 0.1,
                'drug_resistance': 0.7,
                'receptor_binding': 0.8,
                'description': 'Beta/Gamma variant - strong immune escape'
            },
            'L452R': {
                'immune_escape': 0.6,
                'transmissibility': 1.2,
                'virulence': 0.3,
                'drug_resistance': 0.4,
                'receptor_binding': 1.1,
                'description': 'Delta variant - increased transmissibility'
            },
            'K417N': {
                'immune_escape': 0.7,
                'transmissibility': 0.2,
                'virulence': 0.1,
                'drug_resistance': 0.5,
                'receptor_binding': 0.9,
                'description': 'Beta variant - immune escape mutations'
            },
            'S477N': {
                'immune_escape': 0.4,
                'transmissibility': 0.8,
                'virulence': 0.1,
                'drug_resistance': 0.2,
                'receptor_binding': 1.2,
                'description': 'Omicron lineage - receptor binding enhancement'
            }
        }
    
    def predict_mutation_impact(self, mutations: List[str]) -> Dict:
        """
        Predict functional impact of mutation set using Claude 3
        """
        try:
            # Create comprehensive prompt
            prompt = self._create_mutation_analysis_prompt(mutations)
            
            # Call Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "temperature": 0.1,  # Low temperature for consistent scientific analysis
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            ai_analysis = response_body['content'][0]['text']
            
            # Extract structured predictions from AI response
            structured_prediction = self._parse_ai_response(ai_analysis, mutations)
            
            # Validate against known mutations
            validated_prediction = self._validate_with_known_data(structured_prediction, mutations)
            
            logger.info(f"Generated impact prediction for {len(mutations)} mutations")
            
            return validated_prediction
            
        except Exception as e:
            logger.error(f"Error predicting mutation impact: {str(e)}")
            return self._fallback_prediction(mutations)
    
    def _create_mutation_analysis_prompt(self, mutations: List[str]) -> str:
        """
        Create comprehensive prompt for mutation analysis
        """
        mutations_str = ", ".join(mutations)
        
        prompt = f"""
You are a world-class computational virologist analyzing SARS-CoV-2 spike protein mutations. 

Analyze the following spike protein mutations and predict their functional impact:
MUTATIONS: {mutations_str}

For each mutation and the overall mutation set, provide predictions on:

1. IMMUNE ESCAPE POTENTIAL (0.0-1.0 scale):
   - Ability to evade neutralizing antibodies
   - Impact on vaccine effectiveness
   - Monoclonal antibody resistance

2. TRANSMISSIBILITY (0.0-2.0 scale, 1.0 = same as reference):
   - Change in basic reproduction number (R0)
   - ACE2 receptor binding affinity
   - Cell entry efficiency

3. VIRULENCE (0.0-1.0 scale):
   - Disease severity potential
   - Host immune response modulation
   - Pathogenicity changes

4. DRUG RESISTANCE (0.0-1.0 scale):
   - Resistance to antivirals (Paxlovid, Remdesivir)
   - Therapeutic antibody effectiveness

5. STRUCTURAL IMPACT:
   - Protein stability changes
   - Conformational effects
   - Domain-specific impacts

For your analysis, consider:
- Position within functional domains (RBD, NTD, S1/S2, etc.)
- Known mutation effects from scientific literature
- Structural biology implications
- Evolutionary fitness

CRITICAL: Format your response as valid JSON with this exact structure:
{{
  "overall_risk_score": <float 0.0-1.0>,
  "immune_escape": <float 0.0-1.0>,
  "transmissibility": <float 0.0-2.0>,
  "virulence": <float 0.0-1.0>,
  "drug_resistance": <float 0.0-1.0>,
  "receptor_binding": <float 0.0-2.0>,
  "individual_mutations": [
    {{
      "mutation": "<mutation notation>",
      "position": <int>,
      "domain": "<functional domain>",
      "impact_score": <float 0.0-1.0>,
      "mechanism": "<brief impact description>"
    }}
  ],
  "risk_assessment": "<HIGH/MEDIUM/LOW>",
  "key_concerns": ["<concern 1>", "<concern 2>"],
  "confidence": <float 0.0-1.0>,
  "scientific_reasoning": "<detailed explanation>"
}}

Base your predictions on established virology principles and peer-reviewed research.
"""
        
        return prompt
    
    def _parse_ai_response(self, ai_response: str, mutations: List[str]) -> Dict:
        """
        Parse and structure AI response into standardized format
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_response = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['overall_risk_score', 'immune_escape', 'transmissibility', 
                                 'virulence', 'drug_resistance', 'risk_assessment']
                
                for field in required_fields:
                    if field not in parsed_response:
                        logger.warning(f"Missing field {field} in AI response")
                        parsed_response[field] = 0.5  # Default moderate risk
                
                return parsed_response
            else:
                logger.error("No valid JSON found in AI response")
                return self._fallback_prediction(mutations)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response JSON: {str(e)}")
            return self._fallback_prediction(mutations)
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return self._fallback_prediction(mutations)
    
    def _validate_with_known_data(self, prediction: Dict, mutations: List[str]) -> Dict:
        """
        Validate AI predictions against known mutation data
        """
        try:
            known_count = 0
            confidence_adjustment = 0.0
            
            for mutation in mutations:
                if mutation in self.known_mutations:
                    known_count += 1
                    known_data = self.known_mutations[mutation]
                    
                    # Compare predictions with known values (allow ±20% variance)
                    for metric in ['immune_escape', 'transmissibility', 'virulence', 'drug_resistance']:
                        if metric in prediction and metric in known_data:
                            predicted = prediction[metric]
                            known = known_data[metric]
                            
                            variance = abs(predicted - known) / max(known, 0.1)
                            if variance > 0.3:  # >30% difference
                                logger.warning(f"Large variance for {mutation} {metric}: predicted={predicted}, known={known}")
                                # Adjust towards known value
                                prediction[metric] = (predicted + known) / 2
            
            # Adjust confidence based on known mutations
            if known_count > 0:
                knowledge_factor = min(known_count / len(mutations), 1.0)
                prediction['confidence'] = prediction.get('confidence', 0.5) * (0.5 + 0.5 * knowledge_factor)
                prediction['validation_notes'] = f"Validated against {known_count} known mutations"
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error validating predictions: {str(e)}")
            return prediction
    
    def _fallback_prediction(self, mutations: List[str]) -> Dict:
        """
        Provide conservative fallback prediction when AI analysis fails
        """
        return {
            'overall_risk_score': 0.5,
            'immune_escape': 0.3,
            'transmissibility': 1.0,
            'virulence': 0.2,
            'drug_resistance': 0.2,
            'receptor_binding': 1.0,
            'individual_mutations': [
                {
                    'mutation': mut,
                    'position': int(re.search(r'\d+', mut).group()) if re.search(r'\d+', mut) else 0,
                    'domain': 'Unknown',
                    'impact_score': 0.5,
                    'mechanism': 'Fallback prediction - insufficient data'
                } for mut in mutations
            ],
            'risk_assessment': 'MEDIUM',
            'key_concerns': ['Insufficient data for analysis'],
            'confidence': 0.3,
            'scientific_reasoning': 'Fallback prediction due to analysis failure',
            'error': 'AI analysis unavailable'
        }
    
    def calculate_variant_risk_score(self, mutations: List[str], 
                                   cluster_growth: float,
                                   geographic_spread: int) -> Dict:
        """
        Calculate comprehensive variant risk score
        """
        try:
            # Get mutation impact prediction
            mutation_impact = self.predict_mutation_impact(mutations)
            
            # Calculate composite risk score
            mutation_risk = mutation_impact.get('overall_risk_score', 0.5)
            
            # Normalize cluster growth (weekly percentage to 0-1 scale)
            growth_risk = min(cluster_growth / 50.0, 1.0)  # 50% weekly growth = max risk
            
            # Geographic spread risk (number of countries)
            spread_risk = min(geographic_spread / 10.0, 1.0)  # 10+ countries = max risk
            
            # Weighted composite score
            composite_score = (
                0.4 * mutation_risk +
                0.3 * growth_risk +
                0.3 * spread_risk
            )
            
            # Risk classification
            if composite_score >= 0.7:
                risk_level = "HIGH"
                alert_threshold = "RED_ALERT"
            elif composite_score >= 0.5:
                risk_level = "MEDIUM"
                alert_threshold = "AMBER_ALERT"
            else:
                risk_level = "LOW"
                alert_threshold = "MONITORING"
            
            return {
                'composite_risk_score': composite_score,
                'risk_level': risk_level,
                'alert_threshold': alert_threshold,
                'mutation_risk': mutation_risk,
                'growth_risk': growth_risk,
                'spread_risk': spread_risk,
                'mutation_analysis': mutation_impact,
                'recommendation': self._generate_recommendations(risk_level, mutation_impact),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating variant risk score: {str(e)}")
            return {
                'composite_risk_score': 0.5,
                'risk_level': 'MEDIUM',
                'alert_threshold': 'MONITORING',
                'error': str(e)
            }
    
    def _generate_recommendations(self, risk_level: str, mutation_impact: Dict) -> List[str]:
        """
        Generate actionable recommendations based on risk assessment
        """
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Immediate WHO notification required",
                "Enhanced airport screening protocols",
                "Accelerate vaccine strain update evaluation",
                "Increase genomic surveillance frequency"
            ])
        
        if mutation_impact.get('immune_escape', 0) > 0.6:
            recommendations.append("Test vaccine effectiveness against variant")
        
        if mutation_impact.get('drug_resistance', 0) > 0.5:
            recommendations.append("Evaluate antiviral drug effectiveness")
        
        if mutation_impact.get('transmissibility', 1.0) > 1.3:
            recommendations.append("Implement enhanced contact tracing")
        
        return recommendations

def lambda_handler(event, context):
    """
    AWS Lambda handler for mutation impact analysis
    """
    try:
        analyzer = BedrockMutationAnalyzer()
        
        # Extract parameters from event
        mutations = event.get('mutations', [])
        cluster_growth = event.get('cluster_growth', 0.0)
        geographic_spread = event.get('geographic_spread', 1)
        
        if not mutations:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No mutations provided for analysis'})
            }
        
        # Perform risk assessment
        risk_assessment = analyzer.calculate_variant_risk_score(
            mutations, cluster_growth, geographic_spread
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Mutation impact analysis completed',
                'risk_assessment': risk_assessment
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
        'mutations': ['N501Y', 'E484K', 'K417N'],
        'cluster_growth': 25.5,  # 25.5% weekly growth
        'geographic_spread': 3   # Detected in 3 countries
    }
    
    test_context = type('Context', (), {'aws_request_id': 'test-request-id'})()
    
    result = lambda_handler(test_event, test_context)
    print(json.dumps(result, indent=2)) 