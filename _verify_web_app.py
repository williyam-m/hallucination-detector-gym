"""Verify the full web app builds and has expected routes."""
import os
os.environ["ENABLE_WEB_INTERFACE"] = "true"

from server.app import app

routes = [r.path for r in app.routes if hasattr(r, "path")]
print("=== ROUTES ===")
for r in sorted(routes):
    print(f"  {r}")

# Verify key routes exist
assert "/" in routes, "Missing / redirect"
assert "/web" in routes, "Missing /web redirect"
assert "/web/metadata" in routes, "Missing /web/metadata"
assert "/web/reset" in routes, "Missing /web/reset"
assert "/web/step" in routes, "Missing /web/step"
assert "/web/state" in routes, "Missing /web/state"
assert "/health" in routes, "Missing /health"
assert "/reset" in routes, "Missing /reset"
assert "/step" in routes, "Missing /step"

# Verify NO TabbedInterface (our custom UI is the only one)
# The app type should be FastAPI, not a TabbedInterface wrapper
print(f"\nApp type: {type(app).__name__}")
assert type(app).__name__ == "FastAPI"

print("\n✅ All route and UI structure assertions passed!")
