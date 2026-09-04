# Cognito migration review notes

- Authentication is performed only when the user sends approved data.
- The desktop app uses OAuth authorization-code flow with PKCE and does not embed a client secret.
- API Gateway validates Cognito JWTs.
- Lambda handlers enforce `safeharbor-site-<SITE_ID>` group membership on every protected operation.
- Upload completion rejects object keys outside the authenticated site prefix.
- IT should still review token lifetime/refresh handling, user lifecycle and group administration, CloudTrail/CloudWatch auditing, WAF/rate limits, and the remaining direct-AWS Streamlit send path before production use.
