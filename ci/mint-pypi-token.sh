#!/bin/bash

# retrieve the ambient OIDC token
resp=$(curl -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
"$ACTIONS_ID_TOKEN_REQUEST_URL&audience=testpypi")
oidc_token=$(jq -r '.value' <<< "${resp}")

echo "OIDC token: $oidc_token"
echo "Response: $resp"

# exchange the OIDC token for an API token
resp=$(curl -X POST https://test.pypi.org/_/oidc/mint-token -d "{\"token\": \"${oidc_token}\"}")

echo "OIDC token: $oidc_token"
echo "Response: $resp"

api_token=$(jq -r '.token' <<< "${resp}")

# mask the newly minted API token, so that we don't accidentally leak it
echo "::add-mask::${api_token}"

# see the next step in the workflow for an example of using this step output
echo "api-token=${api_token}" >> "${GITHUB_OUTPUT}"
