#!/usr/bin/env python3
import os

import aws_cdk as cdk

from safeharbor_stack import SafeHarborStack


app = cdk.App()

env_name = app.node.try_get_context("env") or os.getenv("SAFEHARBOR_ENV", "dev")
existing_bucket = app.node.try_get_context("existing_bucket") or os.getenv(
    "SAFEHARBOR_DATALAKE_BUCKET_NAME", ""
)
api_key_value = app.node.try_get_context("api_key_value") or os.getenv(
    "SAFEHARBOR_API_KEY", ""
)
deploy_ecs_worker = False
existing_vpc_id = app.node.try_get_context("existing_vpc_id") or os.getenv("SAFEHARBOR_EXISTING_VPC_ID", "vpc-0415713e7d04c25e8")
ecs_cluster_name = app.node.try_get_context("ecs_cluster_name") or os.getenv("SAFEHARBOR_ECS_CLUSTER_NAME", "safeharbor")
redcap_bucket = app.node.try_get_context("redcap_bucket") or os.getenv("SAFEHARBOR_REDCAP_BUCKET", "force-gpt")
redcap_snapshot_key = app.node.try_get_context("redcap_snapshot_key") or os.getenv(
    "SAFEHARBOR_REDCAP_SNAPSHOT_KEY",
    "redcap/lookups/redcap_snapshot.json",
)
redcap_backup_bucket = app.node.try_get_context("redcap_backup_bucket") or os.getenv(
    "REDCAP_BACKUP_BUCKET",
    "redcap-data-backups",
)
redcap_metadata_bucket = app.node.try_get_context("redcap_metadata_bucket") or os.getenv(
    "REDCAP_METADATA_BUCKET",
    redcap_backup_bucket,
)
redcap_metadata_key = app.node.try_get_context("redcap_metadata_key") or os.getenv(
    "REDCAP_METADATA_KEY",
    "force_metadata_dictionary.csv",
)
site_ids_key = app.node.try_get_context("site_ids_key") or os.getenv(
    "SAFEHARBOR_SITE_IDS_KEY",
    "config/site_ids.csv",
)
existing_subnet_ids_raw = app.node.try_get_context("existing_subnet_ids") or os.getenv(
    "SAFEHARBOR_EXISTING_SUBNET_IDS",
    "subnet-053bd4cd306e11350,subnet-05ea74e6f689a16dd",
)
existing_sg_ids_raw = app.node.try_get_context("existing_security_group_ids") or os.getenv(
    "SAFEHARBOR_EXISTING_SECURITY_GROUP_IDS",
    "sg-024b4e46cf832b627,sg-0dfce0150444984e6",
)
existing_subnet_ids = [x.strip() for x in str(existing_subnet_ids_raw).split(",") if x and x.strip()]
existing_security_group_ids = [x.strip() for x in str(existing_sg_ids_raw).split(",") if x and x.strip()]

stack_name = f"SafeHarborStack-{env_name}"
stack_account = os.getenv("CDK_DEFAULT_ACCOUNT") or os.getenv("AWS_ACCOUNT_ID")
stack_region = os.getenv("CDK_DEFAULT_REGION") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
if not stack_account:
    raise RuntimeError("Unable to resolve AWS account. Set CDK_DEFAULT_ACCOUNT or AWS_ACCOUNT_ID before deploy.")

SafeHarborStack(
    app,
    stack_name,
    env=cdk.Environment(account=stack_account, region=stack_region),
    env_name=env_name,
    existing_data_lake_bucket_name=existing_bucket or None,
    api_key_value=api_key_value or None,
    create_api_key=False,
    deploy_ecs_worker=deploy_ecs_worker,
    existing_vpc_id=existing_vpc_id or None,
    ecs_cluster_name=ecs_cluster_name or None,
    redcap_bucket_name=redcap_bucket or None,
    redcap_snapshot_key=redcap_snapshot_key or None,
    redcap_backup_bucket_name=redcap_backup_bucket or None,
    redcap_metadata_bucket_name=redcap_metadata_bucket or None,
    redcap_metadata_key=redcap_metadata_key or None,
    site_ids_key=site_ids_key or None,
    existing_subnet_ids=existing_subnet_ids,
    existing_security_group_ids=existing_security_group_ids,
)

app.synth()
