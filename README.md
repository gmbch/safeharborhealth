# SafeHarborAI Cognito IT Review Bundle

This bundle contains the Cognito-authenticated API-backed send path, CDK resources, site-group authorization handlers, desktop final-tab login flow, and distribution-site source.

Users are assigned to groups named `safeharbor-site-<SITE_ID>`. The API authorizer validates Cognito tokens, and each protected Lambda enforces the requested site against those verified groups.

Excluded: PHI/data exports, reports/images, notebooks, model weights, credentials, CDK context/output, caches, and the Windows executable.
