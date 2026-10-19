# Global Constants
PROJECT_NAME = "VesperFlow"
VERSION = "1.4.0"
BUILD_TYPE = ""  # Leave empty for stable/release versions. Use "Beta" or "Alpha" for testing.

# Derived Constants
FULL_APP_NAME = f"{PROJECT_NAME} v{VERSION}{f' ({BUILD_TYPE})' if BUILD_TYPE else ''}"
BINARY_NAME_BASE = "VesperFlow"  # Clean name for .exe file (no spaces)

# Supported Geographic Search Priors: (lat_min, lat_max, lon_min, lon_max, grid_step_deg)
REGION_PRESETS = {
    # 1. Global Scale
    "global":           (-60.0,  75.0, -180.0,  180.0, 4.0),

    # 2. Continents & Macro-Regions
    "europe":           ( 35.0,  71.0,  -11.0,   40.0, 1.0),
    "north_america":    (  7.0,  72.0, -168.0,  -52.0, 2.0),
    "central_america":  (  7.0,  22.0,  -93.0,  -77.0, 0.5),
    "south_america":    (-56.0,  13.0,  -82.0,  -34.0, 2.0),
    "africa":           (-35.0,  37.0,  -18.0,   52.0, 2.0),
    "east_africa":      (-12.0,  12.0,   29.0,   42.0, 0.75),
    "southern_africa":  (-35.0, -15.0,   11.0,   36.0, 0.75),
    "asia":             (-10.0,  75.0,   26.0,  180.0, 2.5),
    "southeast_asia":   (-11.0,  28.0,   92.0,  141.0, 1.0),
    "oceania":          (-48.0,   0.0,  110.0,  180.0, 2.0),
    "middle_east":      ( 12.0,  42.0,   26.0,   64.0, 1.5),
    "caribbean":        (  9.0,  27.0,  -89.0,  -59.0, 0.5),
    "greenland":        ( 59.0,  84.0,  -74.0,  -11.0, 2.0),

    # 3. European Sub-Regions & Countries
    "central_europe":   ( 45.0,  56.0,    5.0,   20.0, 0.5),
    "germany":          ( 47.2,  55.1,    5.8,   15.1, 0.5),
    "uk":               ( 49.8,  60.9,   -8.6,    1.8, 0.5),
    "iberia":           ( 35.8,  43.8,   -9.6,    4.5, 0.5),
    "france":           ( 41.3,  51.1,   -5.2,    9.6, 0.5),
    "italy":            ( 36.6,  47.1,    6.6,   18.6, 0.5),
    "alps":             ( 44.0,  48.5,    5.5,   16.5, 0.25),
    "scandinavia":      ( 54.5,  71.2,    4.5,   31.5, 0.75),
    "mediterranean":    ( 30.0,  46.0,   -6.0,   36.0, 0.75),

    # 4. Americas, Africa & Asia-Pacific Countries / Biologging Hubs
    "usa_conus":        ( 24.0,  50.0, -125.0,  -66.0, 1.5),
    "canada":           ( 42.0,  75.0, -141.0,  -52.0, 2.0),
    "mexico":           ( 14.5,  32.7, -117.2,  -86.7, 1.0),
    "panama":           (  7.0,   9.7,  -83.1,  -77.1, 0.25),
    "costa_rica":       (  8.0,  11.3,  -86.0,  -82.5, 0.25),
    "amazon":           (-16.0,   5.0,  -79.0,  -44.0, 1.0),
    "brazil":           (-33.8,   5.3,  -74.0,  -34.8, 1.5),
    "kenya_tanzania":   (-11.8,   5.0,   29.5,   41.9, 0.5),
    "south_africa":     (-34.9, -22.1,   16.4,   32.9, 0.5),
    "madagascar":       (-26.0, -11.5,   43.0,   51.0, 0.5),
    "india":            (  6.0,  37.0,   68.0,   98.0, 1.0),
    "indonesia":        (-11.0,   6.0,   94.0,  141.0, 1.0),
    "borneo":           ( -4.2,   7.4,  108.8,  119.3, 0.5),
    "japan":            ( 30.0,  46.0,  128.0,  146.0, 0.5),
    "australia":        (-44.0, -10.0,  112.0,  154.0, 1.5),
    "new_zealand":      (-47.5, -34.0,  166.0,  179.0, 0.5),

    # 5. Islands & Focused Field Study Micro-Regions
    "galapagos":        ( -1.5,   0.7,  -91.8,  -89.0, 0.25),
    "canary_islands":   ( 27.6,  29.5,  -18.2,  -13.4, 0.25),
    "affenwald":        (51.39, 51.40, 10.739, 10.750, 0.005),
}
