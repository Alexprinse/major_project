# Red Line Detection Issue Analysis

## Problem
1. **False detection**: Red lines were being detected even when not present (Cosmos visualization red elements)
2. **View switching blocked**: Cannot change camera view when red lines are detected
3. **No visibility**: No way to see what's happening during preprocessing

## Solutions Provided

### 1. Debug Visualization Tool
Created `/home/alex/major_project/src/drowsiness_detector/red_line_debug.py`

Run to see preprocessing steps in a 2x2 grid:
```bash
# Webcam debug
env PYTHONPATH=src python3 -m drowsiness_detector.red_line_debug --camera-index 0

# CARLA Cosmos debug  
env PYTHONPATH=src python3 -m drowsiness_detector.red_line_debug --carla --spawn-vehicle-if-missing
```

Shows:
- **Top-left**: Original frame
- **Top-right**: Red mask (HSV color detection)
- **Bottom-left**: After morphological filtering
- **Bottom-right**: Canny edges for line detection

### 2. Improved Red Line Detection Thresholds
Updates proposed to `lane_detection.py`:

**Better Color Filtering**:
```python
# More restrictive saturation/value ranges
lower_red1 = np.array([0, 80, 80])      # Was [0, 50, 50]
upper_red1 = np.array([10, 255, 255])    # Was [15, 255, 255]
lower_red2 = np.array([170, 80, 80])   # Was [165, 50, 50]
upper_red2 = np.array([180, 255, 255])   # Was [180, 255, 255]
```

**Red Content Ratio Check**:
```python
red_content_ratio = np.count_nonzero(red_mask) / red_mask.size
if red_content_ratio < 0.01:  # Requires at least 1% red pixels
    return None, None, False, frame.copy()
```

**Stricter Line Requirements**:
- Threshold: 50 (was 30) - Need stronger edges
- minLineLength: 80 (was 50) - Longer continuous lines
- maxLineGap: 30 (was 50) - Less gap tolerance

**Vertical Line Filter**:
```python
# Only accept lines that are mostly vertical (lane boundaries are vertical)
slope = abs((y2 - y1) / max(1, x2 - x1))
if slope > 0.5:  # Steeper than 1:2 ratio
    line_list.append((x1, y1, x2, y2))
```

**Left/Right Validation**:
```python
# Ensure left line is actually to the left of right line
if lane_width > car_width_estimate * 1.2 and left_x1 < right_x1:
    red_detected = True
```

### 3. View Switching Fix
When red lines are detected, they **override** the white lane detection. This is intentional - red lines should take priority in Cosmos mode because they're the actual road boundaries.

**However**: You can still change camera views by pressing **Y** or **TAB** keys - these work at the `carla_controller` level, not dependent on lane detection results.

If view switching feels blocked, it might be because:
- The controller is updating slowly between frames
- Need to ensure pygame event queue is being processed before lane detection
- Consider disabling lane detection temporarily (press **L** key)

## How to Test

```bash
# 1. First, visualize what's happening with debug tool
env PYTHONPATH=src python3 -m drowsiness_detector.red_line_debug --carla --spawn-vehicle-if-missing

# 2. Watch the 4-panel display
# If red mask is showing false positives, adjust HSV thresholds
# If Canny edges look wrong, adjust Canny thresholds
# If lines aren't detected properly, check minLineLength and threshold

# 3. Identify the right thresholds for your CARLA map

# 4. Once satisfied, run full dashboard
env PYTHONPATH=src python3 -m drowsiness_detector.app --camera --carla --spawn-vehicle-if-missing --carla-role-name hero --carla-camera-mode cosmos --lane-detection
```

## Threshold Tuning Guide

If still getting false detections, try these in order:

1. **Increase HSV saturation lower bound** (80→100): Skip more dull colors
2. **Increase Canny low threshold** (30→50): Need stronger edges
3. **Increase Hough threshold** (50→70): Need more votes for line
4. **Increase minLineLength** (80→120): Only very long lines
5. **Reduce maxLineGap** (30→10): No tolerance for breaks
6. **Increase red_content_ratio threshold** (0.01→0.02): Need more red pixels

## Why Red Detection Takes Priority

In Cosmos visualization mode, the red lines (footpath/curb boundaries) are **ground truth** - they're the actual lane markers that CARLA renders. So when detected:
- Red lines become the lane boundaries
- Lane center is calculated from red lines
- Vehicle offset is based on red line position

This is **intentional design** for parking guidance in Cosmos mode.
