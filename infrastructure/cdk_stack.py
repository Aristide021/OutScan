"""
AWS CDK Infrastructure Stack for OutScan Early Warning System
Deploys serverless genomic surveillance architecture
"""
from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as lambda_,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_dynamodb as dynamodb,
    aws_events as events,
    aws_events_targets as targets,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_apigateway as apigateway,
    aws_sns as sns,
    aws_iam as iam,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    RemovalPolicy,
    CfnOutput
)
from constructs import Construct

class OutScanStack(Stack):
    """
    Main CDK stack for OutScan pandemic early warning system
    """
    
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        # Create S3 buckets
        self.create_storage_layer()
        
        # Create DynamoDB tables
        self.create_database_layer()
        
        # Create SNS topics for alerts (before compute layer to provide env vars)
        self.create_notification_layer()
        
        # Create Lambda functions
        self.create_compute_layer()
        
        # Create Step Functions workflow
        self.create_orchestration_layer()
        
        # Create API Gateway
        self.create_api_layer()
        
        # Create monitoring and alarms
        self.create_monitoring_layer()
        
        # Output important ARNs and URLs
        self.create_outputs()
    
    def create_storage_layer(self):
        """Create S3 buckets for genomic data storage"""
        
        # Genomic data bucket with lifecycle policies
        self.genomic_bucket = s3.Bucket(
            self, "GenomicDataBucket",
            bucket_name=f"outscan-genomic-data-{self.account}",
            encryption=s3.BucketEncryption.KMS_MANAGED,
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ArchiveOldSequences",
                    expiration=Duration.days(2555),  # 7 years retention
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(30)
                        ),
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90)
                        )
                    ]
                )
            ]
        )
        
        # Analysis results bucket
        self.analysis_bucket = s3.Bucket(
            self, "AnalysisResultsBucket", 
            bucket_name=f"outscan-analysis-results-{self.account}",
            encryption=s3.BucketEncryption.KMS_MANAGED,
            public_read_access=False
        )
        
        # Public dashboard data bucket
        self.public_bucket = s3.Bucket(
            self, "PublicDashboardBucket",
            bucket_name=f"outscan-public-data-{self.account}",
            encryption=s3.BucketEncryption.S3_MANAGED,
            public_read_access=True,
            block_public_access=s3.BlockPublicAccess(
                block_public_acls=False,
                ignore_public_acls=False,
                block_public_policy=False,
                restrict_public_buckets=False
            ),
            website_index_document="index.html"
        )
    
    def create_database_layer(self):
        """Create DynamoDB tables for variant tracking"""
        
        # Sequence storage table
        self.sequence_table = dynamodb.Table(
            self, "SequenceTable",
            table_name="OutScan-SequenceTable",
            partition_key=dynamodb.Attribute(
                name="sequence_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN
        )
        
        # Variant detection table
        self.variant_table = dynamodb.Table(
            self, "VariantTable",
            table_name="OutScan-VariantTable",
            partition_key=dynamodb.Attribute(
                name="variant_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN
        )
        
        # Alert storage table
        self.alert_table = dynamodb.Table(
            self, "AlertTable",
            table_name="OutScan-AlertTable",
            partition_key=dynamodb.Attribute(
                name="alert_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN
        )
    
    def create_compute_layer(self):
        """Create Lambda functions for genomic processing"""
        
        # Common Lambda execution role
        lambda_role = iam.Role(
            self, "OutScanLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaBasicExecutionRole")
            ],
            inline_policies={
                "OutScanPermissions": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "s3:GetObject",
                                "s3:PutObject", 
                                "s3:DeleteObject",
                                "s3:ListBucket"
                            ],
                            resources=[
                                self.genomic_bucket.bucket_arn,
                                self.genomic_bucket.bucket_arn + "/*",
                                self.analysis_bucket.bucket_arn,
                                self.analysis_bucket.bucket_arn + "/*",
                                self.public_bucket.bucket_arn,
                                self.public_bucket.bucket_arn + "/*"
                            ]
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "dynamodb:GetItem",
                                "dynamodb:PutItem",
                                "dynamodb:UpdateItem",
                                "dynamodb:Query",
                                "dynamodb:Scan",
                                "dynamodb:BatchGetItem",
                                "dynamodb:BatchWriteItem"
                            ],
                            resources=[
                                self.sequence_table.table_arn,
                                self.sequence_table.table_arn + "/*",
                                self.variant_table.table_arn,
                                self.variant_table.table_arn + "/*",
                                self.alert_table.table_arn,
                                self.alert_table.table_arn + "/*"
                            ]
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "bedrock:InvokeModel"
                            ],
                            resources=["*"]
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "sns:Publish"
                            ],
                            resources=["*"]
                        ),
                        iam.PolicyStatement(
                            effect=iam.Effect.ALLOW,
                            actions=[
                                "states:StartExecution"
                            ],
                            resources=["*"]
                        )
                    ]
                )
            }
        )
        
        # GISAID Downloader Lambda
        self.gisaid_lambda = lambda_.Function(
            self, "GISAIDDownloaderFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="gisaid_downloader.lambda_handler",
            code=lambda_.Code.from_asset("../genomic_ingestion"),
            timeout=Duration.minutes(15),
            memory_size=1024,
            role=lambda_role,
            environment={
                "GENOMIC_DATA_BUCKET": self.genomic_bucket.bucket_name
            }
        )
        
        # S3 Trigger Processor Lambda (dependency-free with custom FASTA parser)
        self.s3_processor_lambda = lambda_.Function(
            self, "S3ProcessorFunction", 
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="s3_trigger_processor.lambda_handler",
            code=lambda_.Code.from_asset("../genomic_ingestion", 
                bundling={
                    "image": lambda_.Runtime.PYTHON_3_9.bundling_image,
                    "command": [
                        "bash", "-c",
                        "pip install --no-cache-dir -r requirements.txt -t /asset-output && cp -au . /asset-output"
                    ]
                }
            ),
            timeout=Duration.minutes(15),
            memory_size=2048,
            role=lambda_role,
            environment={
                "GENOMIC_DATA_BUCKET": self.genomic_bucket.bucket_name
            }
        )
        
        # Variant Clustering Engine Lambda (advanced Ward linkage clustering)
        self.clustering_lambda = lambda_.Function(
            self, "ClusteringEngineFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="clustering_engine_advanced.lambda_handler",
            code=lambda_.Code.from_asset("../variant_analysis"),
            timeout=Duration.minutes(15),  # Lambda maximum timeout
            memory_size=1024,  # Increased memory for advanced algorithms
            role=lambda_role,
            environment={
                "MIN_CLUSTER_SIZE": "5",
                "CLUSTER_SELECTION_EPSILON": "0.7",
                "MAX_CLUSTERS": "20"
            }
        )
        
        # Bedrock Inference Lambda
        self.bedrock_lambda = lambda_.Function(
            self, "BedrockInferenceFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="bedrock_inference.lambda_handler", 
            code=lambda_.Code.from_asset("../variant_analysis"),
            timeout=Duration.minutes(10),
            memory_size=1024,
            role=lambda_role,
            environment={
                "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20250229-v1:0"
            }
        )
        
        # WHO Report Generator Lambda
        self.who_report_lambda = lambda_.Function(
            self, "WHOReportFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="who_report_generator.lambda_handler",
            code=lambda_.Code.from_asset("../alerting"),
            timeout=Duration.minutes(5),
            memory_size=512,
            role=lambda_role
        )
        
        # SNS Dispatcher Lambda
        self.sns_dispatcher_lambda = lambda_.Function(
            self, "SNSDispatcherFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="sns_dispatcher.lambda_handler", 
            code=lambda_.Code.from_asset("../alerting"),
            timeout=Duration.minutes(5),
            memory_size=512,
            role=lambda_role,
            environment={
                "WHO_ALERTS_TOPIC_ARN": self.who_alerts_topic.topic_arn,
                "HEALTH_AUTHORITIES_TOPIC_ARN": self.health_authorities_topic.topic_arn,
                "RESEARCH_TOPIC_ARN": self.research_topic.topic_arn,
                "PUBLIC_TOPIC_ARN": self.public_topic.topic_arn,
                "ALERT_TABLE_NAME": self.alert_table.table_name
            }
        )
        
        # S3 trigger for genomic processing
        self.genomic_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(self.s3_processor_lambda),
            s3.NotificationKeyFilter(prefix="genomic-sequences/")
        )
    
    def create_orchestration_layer(self):
        """Create Step Functions for variant analysis workflow"""
        
        # Define workflow steps with error handling
        workflow_failed = sfn.Fail(
            self, "WorkflowFailed",
            comment="Genomic analysis workflow failed"
        )
        
        extract_mutations = tasks.LambdaInvoke(
            self, "ExtractMutations",
            lambda_function=self.s3_processor_lambda,
            output_path="$.Payload",
            result_path="$.mutations"
        ).add_catch(
            errors=["States.TaskFailed", "States.Timeout"],
            next=workflow_failed,
            result_path="$.error"
        ).add_retry(
            errors=["States.TaskFailed"],
            interval=Duration.seconds(30),
            max_attempts=2,
            backoff_rate=2.0
        )
        
        # Clustering analysis step
        clustering_analysis = tasks.LambdaInvoke(
            self, "ClusteringAnalysis",
            lambda_function=self.clustering_lambda,
            output_path="$.Payload",
            result_path="$.clustering_results"
        ).add_catch(
            errors=["States.TaskFailed", "States.Timeout"],
            next=workflow_failed,
            result_path="$.error"
        ).add_retry(
            errors=["States.TaskFailed"],
            interval=Duration.seconds(60),
            max_attempts=2,
            backoff_rate=2.0
        )
        
        bedrock_analysis = tasks.LambdaInvoke(
            self, "BedrockAnalysis", 
            lambda_function=self.bedrock_lambda,
            output_path="$.Payload",
            result_path="$.risk_assessment"
        ).add_catch(
            errors=["States.TaskFailed", "States.Timeout"],
            next=workflow_failed,
            result_path="$.error"
        ).add_retry(
            errors=["States.TaskFailed"],
            interval=Duration.seconds(30),
            max_attempts=3,
            backoff_rate=2.0
        )
        
        generate_who_report = tasks.LambdaInvoke(
            self, "GenerateWHOReport",
            lambda_function=self.who_report_lambda,
            output_path="$.Payload",
            result_path="$.who_report"
        ).add_catch(
            errors=["States.TaskFailed", "States.Timeout"],
            next=workflow_failed,
            result_path="$.error"
        ).add_retry(
            errors=["States.TaskFailed"],
            interval=Duration.seconds(30),
            max_attempts=2,
            backoff_rate=2.0
        )
        
        dispatch_alerts = tasks.LambdaInvoke(
            self, "DispatchAlerts",
            lambda_function=self.sns_dispatcher_lambda,
            output_path="$.Payload"
        ).add_catch(
            errors=["States.TaskFailed", "States.Timeout"],
            next=workflow_failed,
            result_path="$.error"
        ).add_retry(
            errors=["States.TaskFailed"],
            interval=Duration.seconds(30),
            max_attempts=3,
            backoff_rate=2.0
        )
        
        # Risk assessment choice
        risk_choice = sfn.Choice(self, "RiskAssessment")
        
        high_risk_path = generate_who_report.next(dispatch_alerts)
        medium_risk_path = dispatch_alerts
        low_risk_path = sfn.Succeed(self, "LowRiskComplete")
        
        risk_choice.when(
            sfn.Condition.number_greater_than("$.risk_assessment.Payload.composite_risk_score", 0.7),
            high_risk_path
        ).when(
            sfn.Condition.number_greater_than("$.risk_assessment.Payload.composite_risk_score", 0.4),
            medium_risk_path  
        ).otherwise(low_risk_path)
        
        # Define workflow with clustering
        workflow_definition = extract_mutations.next(
            clustering_analysis
        ).next(
            bedrock_analysis
        ).next(risk_choice)
        
        # Create state machine
        self.variant_analysis_sfn = sfn.StateMachine(
            self, "VariantAnalysisWorkflow",
            definition=workflow_definition,
            timeout=Duration.hours(1)
        )
        
        # EventBridge rule for scheduled analysis
        analysis_rule = events.Rule(
            self, "VariantAnalysisSchedule",
            schedule=events.Schedule.rate(Duration.hours(6)),  # Run every 6 hours
            targets=[targets.SfnStateMachine(self.variant_analysis_sfn)]
        )
        
        # EventBridge rule for clustering analysis (more frequent)
        clustering_rule = events.Rule(
            self, "ClusteringAnalysisSchedule",
            schedule=events.Schedule.rate(Duration.hours(12)),  # Run every 12 hours
            targets=[targets.LambdaFunction(self.clustering_lambda)]
        )
    
    def create_notification_layer(self):
        """Create SNS topics for alert distribution"""
        
        # WHO alerts topic
        self.who_alerts_topic = sns.Topic(
            self, "WHOAlertsTopic",
            topic_name="outscan-who-alerts",
            display_name="OutScan WHO Alerts"
        )
        
        # Health authorities topic
        self.health_authorities_topic = sns.Topic(
            self, "HealthAuthoritiesTopic", 
            topic_name="outscan-health-authorities",
            display_name="OutScan Health Authorities"
        )
        
        # Research community topic
        self.research_topic = sns.Topic(
            self, "ResearchTopic",
            topic_name="outscan-research-alerts", 
            display_name="OutScan Research Community"
        )
        
        # Public dashboard topic
        self.public_topic = sns.Topic(
            self, "PublicTopic",
            topic_name="outscan-public-alerts",
            display_name="OutScan Public Alerts"
        )
    
    def create_api_layer(self):
        """Create API Gateway for external integrations"""
        
        # REST API
        self.api = apigateway.RestApi(
            self, "OutScanAPI",
            rest_api_name="OutScan Early Warning API",
            description="API for genomic data submission and variant queries",
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=apigateway.Cors.ALL_ORIGINS,
                allow_methods=apigateway.Cors.ALL_METHODS,
                allow_headers=["Content-Type", "Authorization"]
            )
        )
        
        # API resources
        genomic_resource = self.api.root.add_resource("genomic")
        variants_resource = self.api.root.add_resource("variants")
        alerts_resource = self.api.root.add_resource("alerts")
        
        # Demo API handler with proper CORS
        variant_query_lambda = lambda_.Function(
            self, "VariantQueryFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="api_handler.lambda_handler",
            code=lambda_.Code.from_asset("../api"),
            timeout=Duration.seconds(30),
            environment={
                "VARIANT_TABLE": self.variant_table.table_name
            }
        )
        
        # Grant DynamoDB access to query function
        self.variant_table.grant_read_data(variant_query_lambda)
        
        # API methods
        genomic_resource.add_method(
            "POST",
            apigateway.LambdaIntegration(self.gisaid_lambda),
            api_key_required=True
        )
        
        variants_resource.add_method(
            "GET", 
            apigateway.LambdaIntegration(variant_query_lambda)
        )
        
        # API Key for external integrations
        api_key = self.api.add_api_key(
            "OutScanAPIKey",
            api_key_name="outscan-genomic-submission"
        )
        
        # Usage plan
        usage_plan = self.api.add_usage_plan(
            "OutScanUsagePlan",
            name="OutScan Standard Plan",
            throttle=apigateway.ThrottleSettings(
                rate_limit=1000,
                burst_limit=2000
            ),
            quota=apigateway.QuotaSettings(
                limit=10000,
                period=apigateway.Period.DAY
            )
        )
        
        usage_plan.add_api_key(api_key)
        usage_plan.add_api_stage(
            stage=self.api.deployment_stage
        )
    
    def create_monitoring_layer(self):
        """Create CloudWatch dashboards and alarms"""
        
        # CloudWatch Dashboard
        dashboard = cloudwatch.Dashboard(
            self, "OutScanDashboard",
            dashboard_name="OutScan-Monitoring"
        )
        
        # Add widgets for key metrics
        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Genomic Sequences Processed",
                left=[
                    self.s3_processor_lambda.metric_invocations(),
                    self.s3_processor_lambda.metric_errors()
                ]
            ),
            cloudwatch.GraphWidget(
                title="Variant Analysis Workflows",
                left=[
                    self.variant_analysis_sfn.metric_succeeded(),
                    self.variant_analysis_sfn.metric_failed()
                ]
            )
        )
        
        # Alarms
        high_error_alarm = cloudwatch.Alarm(
            self, "HighErrorRateAlarm",
            metric=self.s3_processor_lambda.metric_errors(
                period=Duration.minutes(5)
            ),
            threshold=10,
            evaluation_periods=2
        )
        
        # SNS topic for operational alerts
        ops_topic = sns.Topic(
            self, "OperationalAlerts",
            topic_name="outscan-operational-alerts"
        )
        
        high_error_alarm.add_alarm_action(
            cw_actions.SnsAction(ops_topic)
        )
    
    def create_outputs(self):
        """Create CloudFormation outputs for important resources"""
        
        CfnOutput(
            self, "GenomicDataBucketName",
            value=self.genomic_bucket.bucket_name,
            description="S3 bucket for genomic data storage"
        )
        
        CfnOutput(
            self, "APIGatewayURL",
            value=self.api.url,
            description="OutScan API Gateway endpoint"
        )
        
        CfnOutput(
            self, "VariantAnalysisWorkflowArn",
            value=self.variant_analysis_sfn.state_machine_arn,
            description="Step Functions workflow for variant analysis"
        )
        
        CfnOutput(
            self, "CloudWatchDashboardURL",
            value=f"https://console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name=OutScan-Monitoring",
            description="CloudWatch monitoring dashboard"
        )