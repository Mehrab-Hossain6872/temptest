# Bounding Box Configuration for Smaller Map Areas

This feature allows you to load only a smaller portion of the map data, which significantly reduces loading time and memory usage.

## How It Works

Instead of loading the entire Noord-Holland region (173MB), you can now specify a bounding box to load only the data within that geographic area.

## Available Areas

Run this command to see all available areas:

```bash
cd backend
python -c "from bbox_config import print_available_areas; print_available_areas()"
```

### Pre-configured Areas:

- **amsterdam_center**: Amsterdam City Center (small, fast loading)
- **amsterdam_metro**: Amsterdam Metropolitan Area (medium size)
- **haarlem**: Haarlem city area
- **utrecht**: Utrecht city center
- **the_hague**: The Hague city center
- **rotterdam**: Rotterdam city center
- **leiden**: Leiden city center
- **delft**: Delft city center
- **gouda**: Gouda city center

## How to Change the Area

### Method 1: Using the Utility Script (Recommended)

```bash
cd backend
python change_area.py amsterdam_center
```

### Method 2: Manual Edit

Edit `backend_main.py` and change this line:

```python
area_name = "amsterdam_center"  # Change this to load different areas
```

## Performance Benefits

- **Faster startup**: Loading time reduced from minutes to seconds
- **Lower memory usage**: Significantly reduced RAM requirements
- **Faster routing**: Smaller graph means faster path calculations
- **Better for development**: Quick iterations and testing

## Custom Bounding Box

To add a custom area, edit `bbox_config.py` and add a new bounding box:

```python
# Your custom area
CUSTOM_AREA = (north, south, east, west)
```

Then add it to the `get_bbox_by_name` function.

## Bounding Box Format

Bounding boxes are defined as tuples: `(north, south, east, west)`
- **north**: Northernmost latitude
- **south**: Southernmost latitude  
- **east**: Easternmost longitude
- **west**: Westernmost longitude

Example: `(52.42, 52.34, 4.95, 4.85)` covers Amsterdam city center.

## After Changing Areas

1. Restart your backend server
2. The new area will be loaded on startup
3. The GraphML cache will be updated for the new area

## Troubleshooting

- If you get "No network found in specified bounding box", try a larger area
- Make sure coordinates are in the correct format (latitude, longitude)
- Check that the bounding box coordinates are valid for the Netherlands 