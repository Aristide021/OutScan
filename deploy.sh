#!/bin/bash

# OutScan Deployment Script
# This script automates the deployment of OutScan to AWS

set -e  # Exit on any error

echo "🚀 Starting OutScan Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check CDK
if ! command -v cdk &> /dev/null; then
    print_status "AWS CDK not found. Installing globally..."
    npm install -g aws-cdk
fi

print_success "Prerequisites check passed!"

# Verify AWS credentials
print_status "Verifying AWS credentials..."
aws sts get-caller-identity > /dev/null 2>&1
if [ $? -eq 0 ]; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION=$(aws configure get region)
    if [ -z "$REGION" ]; then
        REGION="us-east-1"
        print_warning "No default region set, using us-east-1"
    fi
    print_success "AWS credentials verified. Account: $ACCOUNT_ID, Region: $REGION"
else
    print_error "AWS credentials not configured. Please run 'aws configure'"
    exit 1
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
print_success "Python dependencies installed!"

# Check Bedrock model access
print_status "Checking Amazon Bedrock model access..."
python3 -c "
import boto3
try:
    client = boto3.client('bedrock', region_name='$REGION')
    client.list_foundation_models()
    print('✅ Bedrock access confirmed')
except Exception as e:
    print('⚠️  Bedrock access issue:', str(e))
    print('⚠️  You may need to request model access in the AWS Console')
" 2>/dev/null || print_warning "Could not verify Bedrock access. You may need to request model access."

# Bootstrap CDK (if not already done)
print_status "Bootstrapping CDK..."
cd infrastructure
cdk bootstrap aws://$ACCOUNT_ID/$REGION
print_success "CDK bootstrapped!"

# Synthesize the stack
print_status "Synthesizing CDK stack..."
cdk synth
print_success "CDK synthesis complete!"

# Deploy the stack
print_status "Deploying OutScan infrastructure..."
print_warning "This may take 10-15 minutes..."
cdk deploy --require-approval never

if [ $? -eq 0 ]; then
    print_success "🎉 OutScan deployed successfully!"
    
    # Get stack outputs
    print_status "Retrieving deployment information..."
    
    # Get API Gateway URL
    API_URL=$(aws cloudformation describe-stacks \
        --stack-name OutScanStack \
        --query 'Stacks[0].Outputs[?OutputKey==`APIGatewayURL`].OutputValue' \
        --output text 2>/dev/null || echo "Not available")
    
    # Get S3 bucket name
    BUCKET_NAME=$(aws cloudformation describe-stacks \
        --stack-name OutScanStack \
        --query 'Stacks[0].Outputs[?OutputKey==`GenomicDataBucketName`].OutputValue' \
        --output text 2>/dev/null || echo "Not available")
    
    echo ""
    echo "🔗 Deployment Details:"
    echo "===================="
    echo "• API Gateway URL: $API_URL"
    echo "• Genomic Data Bucket: $BUCKET_NAME"
    echo "• Region: $REGION"
    echo "• Account: $ACCOUNT_ID"
    echo ""
    
    print_status "Testing API endpoint..."
    if [ "$API_URL" != "Not available" ]; then
        curl -s "${API_URL}variants" | python3 -m json.tool || print_warning "API test failed"
    fi
    
    echo ""
    print_success "🚀 OutScan is now deployed and ready for your demo!"
    echo ""
    echo "Next steps for your demo:"
    echo "1. Access the CloudWatch dashboard to show real-time monitoring"
    echo "2. Use the API endpoint to demonstrate variant queries"
    echo "3. Upload sample genomic data to trigger the processing pipeline"
    echo "4. Show the SNS topics for alert distribution"
    echo ""
    echo "Demo commands:"
    echo "• Test API: curl ${API_URL}variants"
    echo "• Upload test data: aws s3 cp sample.fasta s3://${BUCKET_NAME}/genomic-sequences/"
    echo "• View dashboard: https://console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards:name=OutScan-Monitoring"
    
else
    print_error "Deployment failed. Check the error messages above."
    exit 1
fi

cd .. 