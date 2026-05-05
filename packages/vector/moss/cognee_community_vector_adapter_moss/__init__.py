import os

# Route Moss traffic through the hosted proxy by default. This lets the
# adapter work from environments that can't reach service.usemoss.dev
# directly (notably Daytona's free-tier sandboxes, whose egress allow-list
# blocks usemoss.dev but whitelists *.vercel.app). setdefault means a
# caller who sets their own URL overrides us - no breakage for anyone
# running Moss on their own infra.
os.environ.setdefault(
    "MOSS_CLOUD_API_MANAGE_URL",
    "https://moss-proxy.vercel.app/v1/manage",
)
os.environ.setdefault(
    "MOSS_CLOUD_QUERY_URL",
    "https://moss-proxy.vercel.app/query",
)

from .moss_adapter import MossAdapter  # noqa: E402

__all__ = ["MossAdapter"]
