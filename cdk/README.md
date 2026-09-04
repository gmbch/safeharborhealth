# SafeHarbor CDK Infra

This CDK app provisions backend resources for the Safe Harbor redaction workflow:

- S3 data lake bucket for redacted uploads
- DynamoDB table for review and upload events
- DynamoDB table for review task lifecycle/current state (`uploaded -> ... -> staged_for_review -> ...`)
- Cognito user pool and hosted UI for authenticated senders
- SQS queue for asynchronous extraction jobs
  - FIFO queue with `session_id`-based message grouping
- Lambda extraction consumer (SQS-triggered fallback when ECS worker is disabled)
- Optional ECS/Fargate extraction worker service (only when account allows networking resource creation)
- Lambda APIs for:
  - pre-signed upload URL generation
  - upload completion logging
  - review event logging
- API Gateway endpoints:
  - `POST /uploads/presign`
  - `POST /uploads/complete`
  - `POST /review-events`
  - `GET /site-ids`
- API Key + Usage Plan (required on protected endpoints, including `GET /site-ids`)

## 1) Install dependencies

```powershell
cd safeharbor_infra
python -m pip install -r requirements.txt
```

## 2) Bootstrap (one-time per account/region)

```powershell
cdk bootstrap
```

## 3) Deploy

```powershell
cdk deploy -c env=dev
```

Optional: use an existing S3 bucket instead of creating a new one.

```powershell
cdk deploy -c env=prod -c existing_bucket=safe-harbor-data-lake
```

Optional: override the S3 key for site IDs CSV used by `GET /site-ids`.

```powershell
cdk deploy -c env=dev -c site_ids_key=config/site_ids.csv
```

## 4) Use outputs

After deploy, copy these CloudFormation outputs into your app config:

- `ApiBaseUrl`
- `PresignUploadUrl`
- `CompleteUploadUrl`
- `ReviewEventUrl`
- `SiteIdsUrl`
- `SiteIdsS3Key`
- `DataLakeBucketName`
- `ReviewTableName`
- `ReviewTasksTableName`
- `ExtractionQueueUrl`
- `ExtractionQueueArn`
- `ExtractionConsumerLambdaName`
- `ExtractionWorkerClusterName` (only when `deploy_ecs_worker=true`)
- `ExtractionWorkerServiceName` (only when `deploy_ecs_worker=true`)
- `ExtractionWorkerLogGroupOutput` (only when `deploy_ecs_worker=true`)

## 5) Cognito groups and desktop configuration

The API uses a Cognito authorizer. Users must be placed in groups named:

```text
safeharbor-site-BCH
safeharbor-site-CHA
```

Set `SAFEHARBOR_SITE_GROUPS=BCH,CHA` during deployment to have CDK create those groups. Assign users through the Cognito console or an approved identity-management workflow. A user may belong to multiple site groups.

The deployment outputs `CognitoUserPoolId`, `CognitoClientId`, and `CognitoHostedUiBaseUrl`. Configure the desktop client with:

```powershell
$env:SAFEHARBOR_COGNITO_CLIENT_ID = "<CognitoClientId>"
$env:SAFEHARBOR_COGNITO_HOSTED_UI = "<CognitoHostedUiBaseUrl>"
$env:SAFEHARBOR_COGNITO_REDIRECT_URI = "http://127.0.0.1:8765/callback"
```

The final Send tab opens Cognito login only when the user is ready to send. The API derives allowed sites from verified group claims; the client cannot grant itself access by changing `site_id`.

End users do not run CDK or set these values. The release build should generate `cognito_config.json` from approved CloudFormation outputs and package it beside or inside the signed executable. The values are client identifiers and endpoints, not credentials; never put a client secret, AWS credential, or access token in this file.

## 6) Desktop app environment variables

Set these before launching `desktop_redactor.py`:

```powershell
$env:SAFEHARBOR_API_BASE_URL = "https://<api-id>.execute-api.<region>.amazonaws.com/<stage>"
```

## 6) Security notes for desktop distribution

- Do not ship AWS credentials to hospital desktops.
- Prefer API Gateway + Lambda + pre-signed URL flow (already provisioned here).
- Cognito access tokens are short-lived bearer credentials and must not be logged or persisted in plaintext.
- Keep Cognito user/group administration separate from application deployment.
- If a fallback key must exist in the app, limit blast radius:
  - enforce throttling/quota with API Gateway/WAF
  - monitor unusual request patterns (CloudWatch + WAF)

Recommended evolution:
- move to short-lived bootstrap auth that mints temporary upload/session tokens instead of static embedded keys.

## 7) Site authorization guardrails

The API authorizer expects Cognito group membership for each site:

- `safeharbor-site-<site_id>` (for example `safeharbor-site-BCH`)

`presign_upload`, `complete_upload`, `review_event`, session handlers, and `get_site_ids` resolve the caller's permitted sites from the verified Cognito token. This prevents clients from submitting arbitrary site acronyms/IDs.

## 8) Async extraction workflow

`complete_upload` now:
1. Writes upload event to `ReviewTable`.
2. Upserts lifecycle row to `ReviewTasksTableName`.
3. Enqueues an extraction job to FIFO `ExtractionQueueUrl` (`MessageGroupId=session_id`).

When `deploy_ecs_worker=false`, CDK deploys an SQS-triggered Lambda consumer that writes staged extraction artifacts.
This fallback works in orgs where SCP blocks VPC/IGW creation.
Fallback worker source is under `lambdas/extraction_consumer/`.

ECS mode:
- Set `-c deploy_ecs_worker=true` (or env `SAFEHARBOR_DEPLOY_ECS_WORKER=true`) to deploy ECS/Fargate worker.
- In ECS mode, the Lambda fallback consumer is not attached to the queue.
- ECS mode is configured to import existing network resources (no new VPC/IGW creation).
- Required context/env values:
  - `existing_vpc_id` / `SAFEHARBOR_EXISTING_VPC_ID`
  - `ecs_cluster_name` / `SAFEHARBOR_ECS_CLUSTER_NAME`
  - `existing_subnet_ids` / `SAFEHARBOR_EXISTING_SUBNET_IDS` (comma-separated)
  - `existing_security_group_ids` / `SAFEHARBOR_EXISTING_SECURITY_GROUP_IDS` (comma-separated)

Current defaults in `app.py` are pre-set so you can run:
```powershell
cdk deploy -c env=dev
```

Default ECS cluster name is `safeharbor` (new cluster in existing VPC), not `gruid`.

Example:
```powershell
cdk deploy -c env=dev ^
  -c deploy_ecs_worker=true ^
  -c existing_vpc_id=vpc-0415713e7d04c25e8 ^
  -c ecs_cluster_name=gruid ^
  -c existing_subnet_ids=subnet-053bd4cd306e11350,subnet-05ea74e6f689a16dd ^
  -c existing_security_group_ids=sg-024b4e46cf832b627,sg-0dfce0150444984e6
```

### Deploy note (important)
Because CDK builds a Docker image asset for the worker, Docker Desktop (or compatible Docker runtime) must be running during `cdk deploy`.

### End-to-end validation
After deployment:
1. Upload from desktop app as usual.
2. Confirm `/uploads/complete` response includes `"queued": true`.
3. Verify task status transitions in `ReviewTasksTableName`:
   - `uploaded` -> `extraction_queued` -> `extraction_in_progress` -> `staged_for_review`
4. Check ECS worker logs in `ExtractionWorkerLogGroup` for:
   - `job_received`
   - `job_staged`

Backfill existing S3 data (CMR + stress_test) into extraction queue:
```powershell
python scripts/backfill_s3_to_extraction_queue.py ^
  --bucket safe-harbor-data-lake-dev ^
  --prefix phi-redaction-uploads/ ^
  --modalities cmr,stress_test ^
  --site-api-keys-table safe-harbor-site-api-keys-dev ^
  --review-tasks-table safe-harbor-review-tasks-dev ^
  --extraction-queue-url <ExtractionQueueUrl> ^
  --api-key-id 747cfe202abe4742a3b634ab43b757ecc0424c594eb240898c33a8a1b2633e5e
```

Dry-run first:
```powershell
python scripts/backfill_s3_to_extraction_queue.py ^
  --bucket safe-harbor-data-lake-dev ^
  --prefix phi-redaction-uploads/ ^
  --modalities cmr,stress_test ^
  --site-api-keys-table safe-harbor-site-api-keys-dev ^
  --review-tasks-table safe-harbor-review-tasks-dev ^
  --extraction-queue-url <ExtractionQueueUrl> ^
  --api-key-id 747cfe202abe4742a3b634ab43b757ecc0424c594eb240898c33a8a1b2633e5e ^
  --dry-run
```

Build REDCap lookup snapshot from `force-gpt/REDCap Export.xlsx`:
```powershell
python scripts/sync_redcap_export_snapshot.py ^
  --bucket force-gpt ^
  --key "REDCap Export.xlsx" ^
  --sheets "CMR,Exercise Stress Test" ^
  --out-key redcap/lookups/redcap_snapshot.json
```

Later modalities can be added by extending `--sheets`:
`Echocardiogram,CT,Catheterization`
