#!/usr/bin/env bash
# Re-pin the upstream OpenAPI specs used by the schema-validation step of the
# integration tests. Run it by hand when you want to validate against newer
# specs; the committed copies are what tests use, so nothing changes until you
# run this and commit the result.
set -euo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# OpenAI: spec lives in their public repo.
openai_url="https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml"

# Anthropic: no spec in their repo. The TypeScript SDK records the Stainless
# spec URL in .stats.yml; read it from there so we follow whatever they ship.
stats_url="https://raw.githubusercontent.com/anthropics/anthropic-sdk-typescript/main/.stats.yml"
anthropic_url="$(curl -fsSL "$stats_url" | awk '/^openapi_spec_url:/ {print $2}')"
if [[ -z "${anthropic_url}" ]]; then
	echo "could not read openapi_spec_url from ${stats_url}" >&2
	exit 1
fi

echo "OpenAI    <- ${openai_url}"
curl -fsSL "$openai_url" -o "${dir}/openai.yaml"

echo "Anthropic <- ${anthropic_url}"
curl -fsSL "$anthropic_url" -o "${dir}/anthropic.json"

echo
echo "Pinned. Record these in README.md and commit:"
sha256sum "${dir}/openai.yaml" "${dir}/anthropic.json"
