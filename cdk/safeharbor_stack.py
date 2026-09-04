from __future__ import annotations

import os
from typing import Optional

from aws_cdk import (
    BundlingOptions,
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_apigateway as apigw,
    aws_cognito as cognito,
    aws_ec2 as ec2,
    aws_dynamodb as ddb,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_lambda as _lambda,
    aws_lambda_event_sources as lambda_event_sources,
    aws_logs as logs,
    aws_sqs as sqs,
    aws_s3 as s3,
)
from constructs import Construct


class SafeHarborStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        env_name: str,
        existing_data_lake_bucket_name: Optional[str] = None,
        redcap_bucket_name: Optional[str] = None,
        redcap_snapshot_key: Optional[str] = None,
        redcap_backup_bucket_name: Optional[str] = None,
        redcap_metadata_bucket_name: Optional[str] = None,
        redcap_metadata_key: Optional[str] = None,
        site_ids_key: Optional[str] = None,
        api_key_value: Optional[str] = None,
        create_api_key: bool = False,
        deploy_ecs_worker: bool = False,
        existing_vpc_id: Optional[str] = None,
        ecs_cluster_name: Optional[str] = None,
        existing_subnet_ids: Optional[list[str]] = None,
        existing_security_group_ids: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        uploads_prefix = "phi-redaction-uploads"

        # -------------------------
        # Data Lake Bucket (create or import)
        # -------------------------
        if existing_data_lake_bucket_name:
            data_lake_bucket = s3.Bucket.from_bucket_name(
                self,
                "DataLakeBucket",
                existing_data_lake_bucket_name,
            )
        else:
            data_lake_bucket = s3.Bucket(
                self,
                "DataLakeBucket",
                bucket_name=f"safe-harbor-data-lake-{env_name}",
                block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
                encryption=s3.BucketEncryption.S3_MANAGED,
                enforce_ssl=True,
                versioned=True,
                removal_policy=RemovalPolicy.RETAIN,
                auto_delete_objects=False,
            )
        redcap_snapshot_bucket = s3.Bucket.from_bucket_name(
            self,
            "RedcapSnapshotBucket",
            redcap_bucket_name or "force-gpt",
        )
        redcap_snapshot_object_key = redcap_snapshot_key or "redcap/lookups/redcap_snapshot.json"
        redcap_backup_bucket = s3.Bucket.from_bucket_name(
            self,
            "RedcapBackupBucket",
            redcap_backup_bucket_name or "redcap-data-backups",
        )
        redcap_metadata_bucket = s3.Bucket.from_bucket_name(
            self,
            "RedcapMetadataBucket",
            redcap_metadata_bucket_name or redcap_backup_bucket.bucket_name,
        )
        redcap_metadata_object_key = redcap_metadata_key or "force_metadata_dictionary.csv"
        site_ids_object_key = site_ids_key or "config/site_ids.csv"

        # Cognito authenticates individual senders. Site authorization comes from
        # groups named safeharbor-site-<site_id> and is enforced in each Lambda.
        user_pool = cognito.UserPool(
            self,
            "SafeHarborUserPool",
            user_pool_name=f"safeharbor-users-{env_name}",
            self_sign_up_enabled=False,
            sign_in_aliases=cognito.SignInAliases(email=True),
            password_policy=cognito.PasswordPolicy(
                min_length=12,
                require_lowercase=True,
                require_uppercase=True,
                require_digits=True,
                require_symbols=True,
            ),
            mfa=cognito.Mfa.REQUIRED,
            mfa_second_factor=cognito.MfaSecondFactor(sms=False, otp=True),
            account_recovery=cognito.AccountRecovery.EMAIL_ONLY,
        )
        callback_urls = [
            value.strip()
            for value in os.getenv(
                "SAFEHARBOR_COGNITO_CALLBACK_URLS",
                "http://127.0.0.1:8765/callback,http://localhost:8765/callback",
            ).split(",")
            if value.strip()
        ]
        domain_prefix = os.getenv(
            "SAFEHARBOR_COGNITO_DOMAIN_PREFIX",
            f"safeharbor-{env_name}-{self.account[-8:]}",
        ).lower()
        user_pool_domain = user_pool.add_domain(
            "SafeHarborUserPoolDomain",
            cognito_domain=cognito.CognitoDomainOptions(domain_prefix=domain_prefix),
        )
        user_pool_client = user_pool.add_client(
            "SafeHarborDesktopClient",
            generate_secret=False,
            o_auth=cognito.OAuthSettings(
                flows=cognito.OAuthFlows(authorization_code_grant=True),
                scopes=[cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL, cognito.OAuthScope.PROFILE],
                callback_urls=callback_urls,
            ),
        )
        configured_sites = [
            value.strip().upper()
            for value in os.getenv("SAFEHARBOR_SITE_GROUPS", "").split(",")
            if value.strip()
        ]
        for index, site_id in enumerate(dict.fromkeys(configured_sites)):
            cognito.CfnUserPoolGroup(
                self,
                f"SafeHarborSiteGroup{index}",
                group_name=f"safeharbor-site-{site_id}",
                user_pool_id=user_pool.user_pool_id,
                description=f"SafeHarbor users assigned to site {site_id}",
            )

        # -------------------------
        # DynamoDB Table for review + upload events
        # -------------------------
        review_table = ddb.Table(
            self,
            "ReviewEventsTable",
            table_name=f"safe-harbor-redaction-review-{env_name}",
            partition_key=ddb.Attribute(name="pk", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="sk", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        review_table.add_global_secondary_index(
            index_name="doc-id-index",
            partition_key=ddb.Attribute(name="doc_id", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="recorded_at_utc", type=ddb.AttributeType.STRING),
            projection_type=ddb.ProjectionType.ALL,
        )

        # -------------------------
        # DynamoDB table for case lifecycle/current state used by web app queue
        # -------------------------
        review_tasks_table = ddb.Table(
            self,
            "ReviewTasksTable",
            table_name=f"safe-harbor-review-tasks-{env_name}",
            partition_key=ddb.Attribute(name="pk", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="sk", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.RETAIN,
            stream=ddb.StreamViewType.NEW_AND_OLD_IMAGES,
        )
        review_tasks_table.add_global_secondary_index(
            index_name="site-status-index",
            partition_key=ddb.Attribute(name="site_id", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="status_updated_utc", type=ddb.AttributeType.STRING),
            projection_type=ddb.ProjectionType.ALL,
        )
        review_tasks_table.add_global_secondary_index(
            index_name="file-id-index",
            partition_key=ddb.Attribute(name="file_id", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="status_updated_utc", type=ddb.AttributeType.STRING),
            projection_type=ddb.ProjectionType.ALL,
        )

        session_table = ddb.Table(
            self,
            "ExtractionSessionsTable",
            table_name=f"safe-harbor-extraction-sessions-{env_name}",
            partition_key=ddb.Attribute(name="pk", type=ddb.AttributeType.STRING),
            sort_key=ddb.Attribute(name="sk", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # -------------------------
        # DynamoDB table mapping API key -> site restrictions
        # -------------------------
        site_api_keys_table = ddb.Table(
            self,
            "SiteApiKeysTable",
            table_name=f"safe-harbor-site-api-keys-{env_name}",
            partition_key=ddb.Attribute(name="api_key_id", type=ddb.AttributeType.STRING),
            billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery=True,
            removal_policy=RemovalPolicy.RETAIN,
        )
        site_api_keys_table.add_global_secondary_index(
            index_name="site-acronym-index",
            partition_key=ddb.Attribute(name="site_acronym", type=ddb.AttributeType.STRING),
            projection_type=ddb.ProjectionType.ALL,
        )

        # -------------------------
        # Queue for asynchronous extraction processing
        # -------------------------
        extraction_dlq = sqs.Queue(
            self,
            "ExtractionJobsDlq",
            queue_name=f"safeharbor-extraction-jobs-dlq-{env_name}.fifo",
            retention_period=Duration.days(14),
            fifo=True,
            content_based_deduplication=True,
            encryption=sqs.QueueEncryption.SQS_MANAGED,
        )
        extraction_queue = sqs.Queue(
            self,
            "ExtractionJobsQueue",
            queue_name=f"safeharbor-extraction-jobs-{env_name}.fifo",
            retention_period=Duration.days(4),
            visibility_timeout=Duration.minutes(15),
            receive_message_wait_time=Duration.seconds(20),
            fifo=True,
            content_based_deduplication=True,
            dead_letter_queue=sqs.DeadLetterQueue(
                queue=extraction_dlq,
                max_receive_count=5,
            ),
            encryption=sqs.QueueEncryption.SQS_MANAGED,
        )

        # -------------------------
        # Extractor consumer Lambda (SQS-triggered fallback when ECS is not enabled)
        # -------------------------
        extraction_consumer_fn = None
        if not deploy_ecs_worker:
            extraction_consumer_fn = _lambda.Function(
                self,
                "ExtractionConsumerLambda",
                function_name=f"safeharbor-extraction-consumer-{env_name}",
                runtime=_lambda.Runtime.PYTHON_3_11,
                handler="lambda_function.lambda_handler",
                code=_lambda.Code.from_asset("lambdas/extraction_consumer"),
                timeout=Duration.minutes(5),
                memory_size=1024,
                log_retention=logs.RetentionDays.ONE_MONTH,
                environment={
                    "ENV": env_name,
                    "DATA_LAKE_BUCKET": data_lake_bucket.bucket_name,
                    "REVIEW_TABLE": review_table.table_name,
                    "REVIEW_TASKS_TABLE": review_tasks_table.table_name,
                    "EXTRACTION_OUTPUT_PREFIX": "extractions",
                    "SAFEHARBOR_REDCAP_BUCKET": redcap_snapshot_bucket.bucket_name,
                    "SAFEHARBOR_REDCAP_SNAPSHOT_KEY": redcap_snapshot_object_key,
                    "REDCAP_CACHE_TTL_SEC": "300",
                },
            )
            extraction_consumer_fn.add_event_source(
                lambda_event_sources.SqsEventSource(
                    extraction_queue,
                    batch_size=1,
                    report_batch_item_failures=True,
                )
            )

        # Optional ECS/Fargate extraction worker (requires ability to create networking resources).
        worker_cluster = None
        worker_log_group = None
        worker_task = None
        worker_subnet_csv = ""
        worker_sg_csv = ""
        if deploy_ecs_worker:
            if not existing_vpc_id:
                raise ValueError("deploy_ecs_worker=true requires existing_vpc_id")
            if not existing_subnet_ids:
                raise ValueError("deploy_ecs_worker=true requires existing_subnet_ids")
            if not existing_security_group_ids:
                raise ValueError("deploy_ecs_worker=true requires existing_security_group_ids")

            worker_vpc = ec2.Vpc.from_lookup(
                self,
                "ExistingWorkerVpc",
                vpc_id=existing_vpc_id,
            )
            worker_security_groups = [
                ec2.SecurityGroup.from_security_group_id(
                    self,
                    f"ImportedWorkerSg{i}",
                    sg_id,
                )
                for i, sg_id in enumerate(existing_security_group_ids)
            ]
            worker_subnets = [
                ec2.Subnet.from_subnet_id(
                    self,
                    f"ImportedWorkerSubnet{i}",
                    subnet_id,
                )
                for i, subnet_id in enumerate(existing_subnet_ids)
            ]
            worker_subnet_csv = ",".join(existing_subnet_ids)
            worker_sg_csv = ",".join(existing_security_group_ids)
            worker_cluster = ecs.Cluster(
                self,
                "ExtractionWorkerCluster",
                cluster_name=ecs_cluster_name or "safeharbor",
                vpc=worker_vpc,
            )
            worker_image_asset = ecr_assets.DockerImageAsset(
                self,
                "ExtractionWorkerImage",
                directory="workers/extraction_worker",
            )
            worker_task = ecs.FargateTaskDefinition(
                self,
                "ExtractionWorkerTaskDef",
                family=f"safeharbor-extraction-worker-{env_name}",
                cpu=512,
                memory_limit_mib=4096,
            )
            worker_log_group = logs.LogGroup(
                self,
                "ExtractionWorkerLogGroup",
                log_group_name=f"/safeharbor/extraction-worker/{env_name}",
                retention=logs.RetentionDays.ONE_MONTH,
                removal_policy=RemovalPolicy.RETAIN,
            )
            worker_task.add_container(
                "Worker",
                image=ecs.ContainerImage.from_docker_image_asset(worker_image_asset),
                logging=ecs.LogDrivers.aws_logs(
                    stream_prefix="worker",
                    log_group=worker_log_group,
                ),
                environment={
                    "REVIEW_TASKS_TABLE": review_tasks_table.table_name,
                    "REVIEW_TABLE": review_table.table_name,
                    "SESSION_TABLE": session_table.table_name,
                    "EXTRACTION_QUEUE_URL": extraction_queue.queue_url,
                    "EXTRACTION_OUTPUT_PREFIX": "extractions",
                    "SESSION_IDLE_MAX_POLLS": "9",
                    "TRAIN_DIR": "/app/train_runtime",
                    "REDCAP_BACKUP_BUCKET": redcap_backup_bucket.bucket_name,
                    "REDCAP_METADATA_BUCKET": redcap_metadata_bucket.bucket_name,
                    "REDCAP_METADATA_KEY": redcap_metadata_object_key,
                    "SAFEHARBOR_REDCAP_BUCKET": redcap_snapshot_bucket.bucket_name,
                    "SAFEHARBOR_REDCAP_SNAPSHOT_KEY": redcap_snapshot_object_key,
                    "REDCAP_CACHE_TTL_SEC": "300",
                },
            )

        # -------------------------
        # Lambda Helper
        # -------------------------
        def make_lambda(name: str, folder: str, *, timeout_sec: int = 30, memory: int = 512) -> _lambda.Function:
            return _lambda.Function(
                self,
                name,
                function_name=f"safeharbor-{folder}-{env_name}",
                runtime=_lambda.Runtime.PYTHON_3_11,
                handler="lambda_function.lambda_handler",
                code=_lambda.Code.from_asset(f"lambdas/{folder}"),
                timeout=Duration.seconds(timeout_sec),
                memory_size=memory,
                log_retention=logs.RetentionDays.ONE_MONTH,
                environment={
                    "ENV": env_name,
                    "DATA_LAKE_BUCKET": data_lake_bucket.bucket_name,
                    "UPLOADS_PREFIX": uploads_prefix,
                    "REVIEW_TABLE": review_table.table_name,
                    "REVIEW_TASKS_TABLE": review_tasks_table.table_name,
                    "SITE_API_KEYS_TABLE": site_api_keys_table.table_name,
                    "EXTRACTION_QUEUE_URL": extraction_queue.queue_url,
                    "SESSION_TABLE": session_table.table_name,
                },
            )

        presign_upload_fn = make_lambda("PresignUploadLambda", "presign_upload", timeout_sec=20, memory=256)
        complete_upload_fn = make_lambda("CompleteUploadLambda", "complete_upload", timeout_sec=20, memory=256)
        review_event_fn = make_lambda("ReviewEventLambda", "review_event", timeout_sec=20, memory=256)
        get_site_ids_fn = _lambda.Function(
            self,
            "GetSiteIdsLambda",
            function_name=f"safeharbor-get-site-ids-{env_name}",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset("lambdas/get_site_ids"),
            timeout=Duration.seconds(20),
            memory_size=256,
            log_retention=logs.RetentionDays.ONE_MONTH,
            environment={
                "ENV": env_name,
                "DATA_LAKE_BUCKET": data_lake_bucket.bucket_name,
                "SITE_IDS_KEY": site_ids_object_key,
            },
        )
        redcap_push_fn = _lambda.Function(
            self,
            "RedcapPushLambda",
            function_name=f"safeharbor-redcap-push-{env_name}",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="lambda_function.lambda_handler",
            code=_lambda.Code.from_asset(
                "lambdas/redcap_push",
                bundling=BundlingOptions(
                    image=_lambda.Runtime.PYTHON_3_11.bundling_image,
                    command=[
                        "bash",
                        "-lc",
                        "python -m pip install -r requirements.txt -t /asset-output && cp -r . /asset-output",
                    ],
                ),
            ),
            timeout=Duration.seconds(60),
            memory_size=512,
            log_retention=logs.RetentionDays.ONE_MONTH,
            environment={
                "ENV": env_name,
                "REVIEW_TABLE": review_table.table_name,
                "REVIEW_TASKS_TABLE": review_tasks_table.table_name,
                "REDCAP_WRITE_SECRET_NAME": "redcap_key_sandbox",
                "REDCAP_WRITE_API_URL": os.getenv("REDCAP_SANDBOX_API_URL", "").strip(),
                "REDCAP_EVENT_NAME": "enrollment_arm_1",
            },
        )
        redcap_push_fn.add_event_source(
            lambda_event_sources.DynamoEventSource(
                review_tasks_table,
                starting_position=_lambda.StartingPosition.LATEST,
                batch_size=5,
                bisect_batch_on_error=True,
                retry_attempts=2,
            )
        )
        session_start_fn = None
        session_close_fn = None
        if deploy_ecs_worker and worker_cluster is not None and worker_task is not None:
            session_start_fn = _lambda.Function(
                self,
                "SessionStartLambda",
                function_name=f"safeharbor-session-start-{env_name}",
                runtime=_lambda.Runtime.PYTHON_3_11,
                handler="lambda_function.lambda_handler",
                code=_lambda.Code.from_asset("lambdas/session_start"),
                timeout=Duration.seconds(30),
                memory_size=512,
                log_retention=logs.RetentionDays.ONE_MONTH,
                environment={
                    "ENV": env_name,
                    "SITE_API_KEYS_TABLE": site_api_keys_table.table_name,
                    "SESSION_TABLE": session_table.table_name,
                    "ECS_CLUSTER_NAME": worker_cluster.cluster_name,
                    "ECS_TASK_DEF_ARN": worker_task.task_definition_arn,
                    "ECS_CONTAINER_NAME": "Worker",
                    "ECS_SUBNET_IDS": worker_subnet_csv,
                    "ECS_SECURITY_GROUP_IDS": worker_sg_csv,
                    "SESSION_IDLE_MAX_POLLS": "9",
                },
            )
            session_close_fn = _lambda.Function(
                self,
                "SessionCloseLambda",
                function_name=f"safeharbor-session-close-{env_name}",
                runtime=_lambda.Runtime.PYTHON_3_11,
                handler="lambda_function.lambda_handler",
                code=_lambda.Code.from_asset("lambdas/session_close"),
                timeout=Duration.seconds(20),
                memory_size=256,
                log_retention=logs.RetentionDays.ONE_MONTH,
                environment={
                    "ENV": env_name,
                    "SITE_API_KEYS_TABLE": site_api_keys_table.table_name,
                    "SESSION_TABLE": session_table.table_name,
                },
            )

        # permissions
        data_lake_bucket.grant_put(presign_upload_fn)
        data_lake_bucket.grant_read(complete_upload_fn)
        review_table.grant_read_write_data(complete_upload_fn)
        review_table.grant_read_write_data(review_event_fn)
        data_lake_bucket.grant_read(get_site_ids_fn, site_ids_object_key)
        review_table.grant_read_write_data(redcap_push_fn)
        review_tasks_table.grant_read_write_data(complete_upload_fn)
        review_tasks_table.grant_read_write_data(redcap_push_fn)
        site_api_keys_table.grant_read_data(presign_upload_fn)
        site_api_keys_table.grant_read_data(complete_upload_fn)
        site_api_keys_table.grant_read_data(review_event_fn)
        session_table.grant_read_write_data(complete_upload_fn)
        extraction_queue.grant_send_messages(complete_upload_fn)
        complete_upload_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sqs:SendMessage", "sqs:GetQueueAttributes"],
                resources=[
                    f"arn:aws:sqs:{self.region}:{self.account}:safeharbor-s-*.fifo",
                ],
            )
            )
        if extraction_consumer_fn is not None:
            extraction_queue.grant_consume_messages(extraction_consumer_fn)
            review_table.grant_read_write_data(extraction_consumer_fn)
            review_tasks_table.grant_read_write_data(extraction_consumer_fn)
            data_lake_bucket.grant_read_write(extraction_consumer_fn)
            redcap_snapshot_bucket.grant_read(extraction_consumer_fn)
        if worker_task is not None:
            review_table.grant_read_write_data(worker_task.task_role)
            review_tasks_table.grant_read_write_data(worker_task.task_role)
            session_table.grant_read_write_data(worker_task.task_role)
            extraction_queue.grant_consume_messages(worker_task.task_role)
            data_lake_bucket.grant_read_write(worker_task.task_role)
            redcap_snapshot_bucket.grant_read(worker_task.task_role)
            redcap_backup_bucket.grant_read(worker_task.task_role)
            redcap_metadata_bucket.grant_read(worker_task.task_role)
            worker_task.task_role.add_to_principal_policy(
                iam.PolicyStatement(
                    actions=[
                        "secretsmanager:GetSecretValue",
                        "secretsmanager:DescribeSecret",
                    ],
                    resources=[
                        f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:azure-api*",
                    ],
                )
            )
            worker_task.task_role.add_to_principal_policy(
                iam.PolicyStatement(
                    actions=["kms:Decrypt"],
                    resources=["*"],
                    conditions={
                        "StringEquals": {
                            "kms:ViaService": f"secretsmanager.{self.region}.amazonaws.com",
                        }
                    },
                )
            )
            worker_task.task_role.add_to_principal_policy(
                iam.PolicyStatement(
                    actions=[
                        "sqs:ReceiveMessage",
                        "sqs:DeleteMessage",
                        "sqs:ChangeMessageVisibility",
                        "sqs:GetQueueAttributes",
                        "sqs:GetQueueUrl",
                        "sqs:DeleteQueue",
                    ],
                    resources=[
                        f"arn:aws:sqs:{self.region}:{self.account}:safeharbor-s-*.fifo",
                    ],
                )
            )
        if session_start_fn is not None:
            site_api_keys_table.grant_read_data(session_start_fn)
            session_table.grant_read_write_data(session_start_fn)
            session_start_fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=[
                        "sqs:CreateQueue",
                        "sqs:GetQueueUrl",
                        "sqs:GetQueueAttributes",
                        "sqs:SetQueueAttributes",
                    ],
                    resources=["*"],
                )
            )
            session_start_fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["sqs:TagQueue", "sqs:UntagQueue"],
                    resources=[
                        f"arn:aws:sqs:{self.region}:{self.account}:safeharbor-s-*.fifo",
                    ],
                )
            )
            session_start_fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["ecs:RunTask"],
                    resources=[worker_task.task_definition_arn],
                )
            )
            session_start_fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["ecs:RunTask"],
                    resources=["*"],
                    conditions={
                        "ArnEquals": {
                            "ecs:cluster": worker_cluster.cluster_arn,
                        }
                    },
                )
            )
            session_start_fn.add_to_role_policy(
                iam.PolicyStatement(
                    actions=["iam:PassRole"],
                    resources=[
                        worker_task.task_role.role_arn,
                        worker_task.execution_role.role_arn if worker_task.execution_role is not None else worker_task.task_role.role_arn,
                    ],
                )
            )
        if session_close_fn is not None:
            site_api_keys_table.grant_read_data(session_close_fn)
            session_table.grant_read_write_data(session_close_fn)
        redcap_push_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:DescribeSecret",
                ],
                resources=[
                    f"arn:aws:secretsmanager:{self.region}:{self.account}:secret:redcap_key_sandbox*",
                ],
            )
        )
        redcap_push_fn.add_to_role_policy(
            iam.PolicyStatement(
                actions=["kms:Decrypt"],
                resources=["*"],
                conditions={
                    "StringEquals": {
                        "kms:ViaService": f"secretsmanager.{self.region}.amazonaws.com",
                    }
                },
            )
        )

        # -------------------------
        # API Gateway
        # -------------------------
        api = apigw.RestApi(
            self,
            "SafeHarborApi",
            rest_api_name=f"safeharbor-api-{env_name}",
            deploy_options=apigw.StageOptions(stage_name=env_name),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
                allow_headers=["Content-Type", "Authorization"],
            ),
        )
        authorizer = apigw.CognitoUserPoolsAuthorizer(
            self,
            "SafeHarborCognitoAuthorizer",
            cognito_user_pools=[user_pool],
        )

        uploads = api.root.add_resource("uploads")
        presign = uploads.add_resource("presign")
        complete = uploads.add_resource("complete")
        session_resource = uploads.add_resource("session")
        session_start_resource = session_resource.add_resource("start")
        session_close_resource = session_resource.add_resource("close")
        review_events = api.root.add_resource("review-events")
        site_ids_resource = api.root.add_resource("site-ids")

        # Require API key for all write endpoints (lightweight auth gate).
        presign.add_method(
            "POST",
            apigw.LambdaIntegration(presign_upload_fn),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=authorizer,
        )
        complete.add_method(
            "POST",
            apigw.LambdaIntegration(complete_upload_fn),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=authorizer,
        )
        if session_start_fn is not None:
            session_start_resource.add_method(
                "POST",
                apigw.LambdaIntegration(session_start_fn),
                authorization_type=apigw.AuthorizationType.COGNITO,
                authorizer=authorizer,
            )
        if session_close_fn is not None:
            session_close_resource.add_method(
                "POST",
                apigw.LambdaIntegration(session_close_fn),
                authorization_type=apigw.AuthorizationType.COGNITO,
                authorizer=authorizer,
            )
        review_events.add_method(
            "POST",
            apigw.LambdaIntegration(review_event_fn),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=authorizer,
        )
        site_ids_resource.add_method(
            "GET",
            apigw.LambdaIntegration(get_site_ids_fn),
            authorization_type=apigw.AuthorizationType.COGNITO,
            authorizer=authorizer,
        )

        usage_plan = api.add_usage_plan(
            "SafeHarborUsagePlan",
            name=f"safeharbor-usage-plan-{env_name}",
            throttle=apigw.ThrottleSettings(
                rate_limit=50,
                burst_limit=100,
            ),
            quota=apigw.QuotaSettings(
                limit=200000,
                period=apigw.Period.MONTH,
            ),
        )
        usage_plan.add_api_stage(stage=api.deployment_stage)

        api_key = None
        if create_api_key:
            # Avoid fixed-name collisions across re-deployments.
            api_key = api.add_api_key(
                "SafeHarborApiKey",
                value=api_key_value,
            )
            usage_plan.add_api_key(api_key)

        # -------------------------
        # Outputs
        # -------------------------
        CfnOutput(self, "ApiBaseUrl", value=api.url, description="Base API URL")
        CfnOutput(self, "CognitoUserPoolId", value=user_pool.user_pool_id)
        CfnOutput(self, "CognitoClientId", value=user_pool_client.user_pool_client_id)
        CfnOutput(
            self,
            "CognitoHostedUiBaseUrl",
            value=f"https://{user_pool_domain.domain_name}.auth.{self.region}.amazoncognito.com",
        )
        CfnOutput(
            self,
            "PresignUploadUrl",
            value=f"{api.url}uploads/presign",
            description="POST endpoint that returns pre-signed S3 upload URL",
        )
        CfnOutput(
            self,
            "CompleteUploadUrl",
            value=f"{api.url}uploads/complete",
            description="POST endpoint to mark upload completion in DynamoDB",
        )
        CfnOutput(
            self,
            "ReviewEventUrl",
            value=f"{api.url}review-events",
            description="POST endpoint to record review event metadata",
        )
        CfnOutput(
            self,
            "SiteIdsUrl",
            value=f"{api.url}site-ids",
            description="GET endpoint that returns allowed Site IDs",
        )
        CfnOutput(
            self,
            "SiteIdsS3Key",
            value=site_ids_object_key,
            description="S3 object key used by Site IDs API",
        )
        CfnOutput(
            self,
            "DataLakeBucketName",
            value=data_lake_bucket.bucket_name,
            description="S3 data lake bucket for redacted report uploads",
        )
        CfnOutput(
            self,
            "ReviewTableName",
            value=review_table.table_name,
            description="DynamoDB table for review and upload events",
        )
        CfnOutput(
            self,
            "ReviewTasksTableName",
            value=review_tasks_table.table_name,
            description="DynamoDB table for case lifecycle and current staged state",
        )
        CfnOutput(
            self,
            "RedcapPushLambdaName",
            value=redcap_push_fn.function_name,
            description="Lambda function that pushes finalized review tasks to sandbox REDCap",
        )
        CfnOutput(
            self,
            "SiteApiKeysTableName",
            value=site_api_keys_table.table_name,
            description="DynamoDB mapping from API key id to site_id/site_acronym",
        )
        CfnOutput(
            self,
            "ExtractionSessionsTableName",
            value=session_table.table_name,
            description="DynamoDB table tracking upload sessions and session-scoped queue/task state",
        )
        CfnOutput(
            self,
            "ExtractionQueueUrl",
            value=extraction_queue.queue_url,
            description="SQS queue URL for async extraction jobs",
        )
        CfnOutput(
            self,
            "ExtractionQueueArn",
            value=extraction_queue.queue_arn,
            description="SQS queue ARN for async extraction jobs",
        )
        if extraction_consumer_fn is not None:
            CfnOutput(
                self,
                "ExtractionConsumerLambdaName",
                value=extraction_consumer_fn.function_name,
                description="Lambda function consuming extraction queue",
            )
        if worker_cluster is not None and worker_log_group is not None and worker_task is not None:
            CfnOutput(
                self,
                "ExtractionWorkerClusterName",
                value=worker_cluster.cluster_name,
                description="ECS cluster name for extraction worker run-task jobs",
            )
            CfnOutput(
                self,
                "ExtractionWorkerTaskDefArn",
                value=worker_task.task_definition_arn,
                description="ECS task definition ARN for session-scoped extraction jobs",
            )
            CfnOutput(
                self,
                "ExtractionWorkerLogGroupOutput",
                value=worker_log_group.log_group_name,
                description="CloudWatch log group for extraction worker",
            )
        if session_start_fn is not None:
            CfnOutput(
                self,
                "SessionStartUrl",
                value=f"{api.url}uploads/session/start",
                description="POST endpoint to start an upload session, queue, and ECS task",
            )
        if session_close_fn is not None:
            CfnOutput(
                self,
                "SessionCloseUrl",
                value=f"{api.url}uploads/session/close",
                description="POST endpoint to mark an upload session close_requested",
            )
        if api_key is not None:
            CfnOutput(
                self,
                "ApiKeyId",
                value=api_key.key_id,
                description="API key id for the Safe Harbor usage plan",
            )
