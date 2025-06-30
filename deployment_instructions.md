# OutScan Deployment Instructions

This guide provides step-by-step instructions for deploying the OutScan Pandemic Variant Early-Warning System to AWS.

## Prerequisites

### Required Tools
- AWS CLI v2.x configured with administrative permissions
- Python 3.9 or higher
- Node.js 18+ and npm (for AWS CDK)
- Docker Desktop (for local testing)
- Git

### AWS Account Setup
```bash
# Configure AWS CLI
aws configure
# Enter your Access Key ID, Secret Access Key, Default region (us-east-1), and output format (json)

# Verify configuration
aws sts get-caller-identity
```

### Required AWS Services
Ensure the following services are available in your target region:
- AWS Lambda
- Amazon DynamoDB  
- Amazon S3
- Amazon Bedrock (Claude models)
- Amazon SNS
- AWS Step Functions
- Amazon API Gateway
- Amazon EventBridge

## Step 1: Clone and Setup Repository

```bash
# Clone repository
git clone https://github.com/your-org/outscan.git
cd outscan

# Create Python virtual environment
python -m venv outscan-env
source outscan-env/bin/activate  # On Windows: outscan-env\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

## Step 2: Infrastructure Deployment

### Install AWS CDK
```bash
npm install -g aws-cdk@latest
cdk --version
```

### Bootstrap CDK (First-time setup)
```bash
cdk bootstrap aws://YOUR-ACCOUNT-ID/us-east-1
```

### Deploy Core Infrastructure
```bash
cd infrastructure

# Install CDK dependencies
npm install

# Review what will be deployed
cdk diff OutScanStack

# Deploy the stack
cdk deploy OutScanStack
```

Expected resources created:
- 3 S3 buckets (genomic-data, analysis-results, public-data)
- 3 DynamoDB tables (VariantClusters, MutationLibrary, AlertHistory)
- 5 Lambda functions
- 1 Step Functions state machine
- 4 SNS topics
- API Gateway with usage plan
- CloudWatch dashboard and alarms

## Step 3: Configure Environment Variables

### Lambda Function Configuration
```bash
# Update Lambda environment variables
aws lambda update-function-configuration \
  --function-name OutScan-GISAIDDownloaderFunction \
  --environment Variables='{
    "GISAID_USERNAME":"your_gisaid_username",
    "GISAID_PASSWORD":"your_gisaid_password", 
    "GENOMIC_DATA_BUCKET":"outscan-genomic-data"
  }'

# Configure Bedrock permissions
aws lambda update-function-configuration \
  --function-name OutScan-BedrockInferenceFunction \
  --environment Variables='{
    "BEDROCK_MODEL_ID":"anthropic.claude-3-sonnet-20250229-v1:0"
  }'
```

### SNS Topic Configuration
```bash
# Get SNS topic ARNs
aws sns list-topics --query 'Topics[?contains(TopicArn, `outscan`)]'

# Subscribe health authorities to alerts
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:YOUR-ACCOUNT:outscan-health-authorities \
  --protocol email \
  --notification-endpoint health-team@yourorganization.gov
```

## Step 4: Enable Amazon Bedrock Models

### Request Model Access
1. Open AWS Console → Amazon Bedrock
2. Navigate to Model Access
3. Request access to:
   - Anthropic Claude 3 Sonnet
   - Anthropic Claude 3 Haiku
4. Wait for approval (usually 1-2 business days)

### Test Model Access
```bash
# Test Bedrock access
python -c "
import boto3
client = boto3.client('bedrock-runtime')
response = client.invoke_model(
    modelId='anthropic.claude-3-haiku-20250307-v1:0',
    body='{\"anthropic_version\": \"bedrock-2023-05-31\", \"max_tokens\": 100, \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'
)
print('Bedrock access confirmed')
"
```

## Step 5: Data Source Integration

### GISAID API Setup
1. Register for GISAID account at https://gisaid.org
2. Request API access through GISAID support
3. Store credentials in AWS Secrets Manager:

```bash
aws secretsmanager create-secret \
  --name outscan/gisaid-credentials \
  --description "GISAID API credentials for OutScan" \
  --secret-string '{
    "username": "your_gisaid_username",
    "password": "your_gisaid_password"
  }'
```

### EventBridge Scheduling
```bash
# Create EventBridge rule for periodic genomic data collection
aws events put-rule \
  --name outscan-data-collection \
  --schedule-expression "rate(6 hours)" \
  --description "Trigger OutScan genomic data collection every 6 hours"

# Add Lambda target
aws events put-targets \
  --rule outscan-data-collection \
  --targets "Id"="1","Arn"="arn:aws:lambda:us-east-1:YOUR-ACCOUNT:function:OutScan-GISAIDDownloaderFunction"
```

## Step 6: Validation and Testing

### Run Historical Validation
```bash
# Test system against historical variants
cd validation
python historical_simulator.py

# Expected output:
# Alpha: 23 days lead time
# Delta: 54 days lead time  
# Omicron: 42 days lead time
```

### Performance Stress Test
```bash
# Run stress test (start small)
python stress_test_workload.py

# Monitor CloudWatch metrics during test
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=OutScan-S3ProcessorFunction \
  --start-time 2025-06-01T00:00:00Z \
--end-time 2025-06-01T01:00:00Z \
  --period 300 \
  --statistics Sum
```

### Test Alert System
```bash
# Send test alert
aws lambda invoke \
  --function-name OutScan-SNSDispatcherFunction \
  --payload '{
    "alert_data": {
      "alert_level": "AMBER_ALERT",
      "variant_info": {
        "variant_id": "TEST-001",
        "key_mutations": ["N501Y", "E484K"],
        "countries_detected": ["United States", "Canada"],
        "growth_rate": 25.0
      },
      "risk_assessment": {
        "composite_risk_score": 0.65
      }
    }
  }' \
  test-response.json
```

## Step 7: Monitoring Setup

### CloudWatch Dashboard
The CDK deployment creates a monitoring dashboard. Access it via:
1. AWS Console → CloudWatch → Dashboards
2. Select "OutScan-Monitoring"
3. Pin important metrics

### Operational Alarms
```bash
# Create high error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "OutScan-HighErrorRate" \
  --alarm-description "Alert when Lambda error rate exceeds 5%" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 10 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2
```

### Cost Monitoring
```bash
# Set up cost alerts
aws budgets create-budget \
  --account-id YOUR-ACCOUNT-ID \
  --budget '{
    "BudgetName": "OutScan-Monthly-Budget",
    "BudgetLimit": {
      "Amount": "1000",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'
```

## Step 8: Security Hardening

### Enable VPC Endpoints
```bash
# Create VPC endpoint for S3
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.us-east-1.s3 \
  --route-table-ids rtb-12345678
```

### Configure WAF for API Gateway
```bash
# Create WAF web ACL
aws wafv2 create-web-acl \
  --scope REGIONAL \
  --default-action Allow={} \
  --name OutScanAPIProtection \
  --description "WAF protection for OutScan API"
```

### Enable CloudTrail
```bash
# Create CloudTrail for audit logging
aws cloudtrail create-trail \
  --name outscan-audit-trail \
  --s3-bucket-name outscan-audit-logs \
  --include-global-service-events \
  --is-multi-region-trail
```

## Step 9: Production Readiness

### Backup Configuration
```bash
# Enable DynamoDB Point-in-Time Recovery
aws dynamodb update-continuous-backups \
  --table-name VariantClusters \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true

aws dynamodb update-continuous-backups \
  --table-name MutationLibrary \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true

aws dynamodb update-continuous-backups \
  --table-name AlertHistory \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true
```

### Cross-Region Replication
```bash
# Set up S3 cross-region replication
aws s3api put-bucket-replication \
  --bucket outscan-genomic-data \
  --replication-configuration file://replication-config.json
```

### Data Lifecycle Policies
```bash
# Configure S3 lifecycle for cost optimization
aws s3api put-bucket-lifecycle-configuration \
  --bucket outscan-genomic-data \
  --lifecycle-configuration file://lifecycle-config.json
```

## Step 10: Go-Live Checklist

### Pre-Launch Validation
- [ ] All Lambda functions deployed and tested
- [ ] DynamoDB tables created with proper indexes
- [ ] S3 buckets configured with lifecycle policies
- [ ] SNS topics set up with appropriate subscriptions
- [ ] Bedrock model access approved and tested
- [ ] API Gateway deployed with rate limiting
- [ ] CloudWatch monitoring and alarms active
- [ ] Historical validation shows >35 day lead time
- [ ] Stress test handles target load

### Documentation
- [ ] API documentation published
- [ ] Runbooks created for operations team
- [ ] Incident response procedures documented
- [ ] User guides for health authorities
- [ ] Contact lists for escalation

### Security Review
- [ ] IAM policies follow least privilege
- [ ] Data encryption enabled everywhere
- [ ] Network security groups configured
- [ ] Audit logging enabled
- [ ] Penetration testing completed

## Troubleshooting

### Common Issues

**1. CDK Bootstrap Errors**
```bash
# If bootstrap fails, try explicit region
cdk bootstrap --region us-east-1
```

**2. Lambda Permission Errors**
```bash
# Check Lambda execution role
aws iam get-role --role-name OutScanLambdaRole
```

**3. Bedrock Access Denied**
```bash
# Verify model access
aws bedrock list-foundation-models --region us-east-1
```

**4. DynamoDB Throttling**
```bash
# Check table metrics
aws dynamodb describe-table --table-name VariantClusters
```

### Performance Optimization

**1. Lambda Cold Starts**
```bash
# Enable provisioned concurrency for critical functions
aws lambda put-provisioned-concurrency-config \
  --function-name OutScan-BedrockInferenceFunction \
  --qualifier $LATEST \
  --provisioned-concurrency-settings ProvisionedConcurrencyUnits=10
```

**2. DynamoDB Performance**
```bash
# Enable auto-scaling
aws application-autoscaling register-scalable-target \
  --service-namespace dynamodb \
  --resource-id table/VariantClusters \
  --scalable-dimension dynamodb:table:ReadCapacityUnits \
  --min-capacity 5 \
  --max-capacity 4000
```

## Support

For deployment issues:
1. Check CloudWatch logs for error messages
2. Review CDK deployment outputs
3. Validate AWS service quotas
4. Contact support: deployment-support@outscan.health.gov

## Next Steps

After successful deployment:
1. **Data Integration**: Connect real genomic data sources
2. **Health Authority Onboarding**: Add official notification endpoints  
3. **International Expansion**: Deploy to additional AWS regions
4. **Performance Tuning**: Optimize based on actual usage patterns

---