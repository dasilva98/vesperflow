# Global Constants
PROJECT_NAME = "VesperFLow"
VERSION = "1.3.3"
BUILD_TYPE = (
    ""  # Leave empty for stable/release versions. Use "Beta" or "Alpha" for testing.
)

# Derived Constants
FULL_APP_NAME = f"{PROJECT_NAME} v{VERSION}{f' ({BUILD_TYPE})' if BUILD_TYPE else ''}"

BINARY_NAME_BASE = "VesperFlow"  # Clean name for .exe file (no spaces)
