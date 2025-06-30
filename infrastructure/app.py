#!/usr/bin/env python3
"""
OutScan CDK Application Entry Point
"""
import aws_cdk as cdk
from cdk_stack import OutScanStack

app = cdk.App()

# Deploy the OutScan stack
OutScanStack(
    app, 
    "OutScanStack",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1"
    ),
    description="OutScan Pandemic Variant Early-Warning System"
)

app.synth() 