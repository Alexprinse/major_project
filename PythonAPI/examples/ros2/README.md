# ROS2 Native Example

This example demonstrates how to utilize the ROS 2 native interface in CARLA.

## Prerequisites

To run this example, ensure `docker` is installed in your system, which is used to launch an instance of `rviz` for visualizing sensor data.


## Usage

### Step 1: Start the CARLA Simulator with ROS2 enabled
Launch the CARLA simulator with the ROS 2 integration enabled:

```bash
# If running a package:
./CarlaUE4.sh --ros2

# If running the editor:
make launch ARGS="--ros2 --editor-flags='--ros2'"
```

### Step 2: Run the ROS2 Example

Execute the ROS 2 example script:

```bash
python3 ros2_native.py --file stack.json
```

* The `stack.json` file defines the sensor configuration.
* You can edit this file to adjust the sensor setup according to your requirements.


### Step 3: Run RViz to Visualize Sensor Data

Start `rviz` to visualize the sensor output from CARLA:

> [!NOTE]
Docker must be installed on your system to complete this step.

```bash
./run_rviz.sh
```

If you run `rviz2` directly (outside `run_rviz.sh`), use this workaround command:

```bash
rviz2 --ros-args --remap /tf:=/tf_ignored --remap /tf_static:=/tf_static_ignored
```

This avoids `TF_NO_FRAME_ID` spam caused by malformed TF messages from CARLA native ROS2 publishing in some 0.9.16 setups.

## Multi-Sensor Setup (Front + Semantic)

The default `stack.json` in this folder is configured with an `ego_vehicle` and multiple sensors, including:

- Front RGB camera (`rgb_front`)
- Front depth camera (`depth_front`)
- Front semantic segmentation camera (`semantic_segmentation_front`)
- Rear/chase camera view (`ego_view`)
- LiDAR (`lidar`) and semantic LiDAR (`semantic_lidar`)
- Radar (`radar_front`), GNSS, IMU, collision, lane invasion

After launching the script, list ROS 2 topics and filter CARLA streams:

```bash
ros2 topic list | grep -E "^/carla/"
```

Typical topics for front and semantic streams include:

- `/carla/ego_vehicle/rgb_front/image`
- `/carla/ego_vehicle/rgb_front/camera_info`
- `/carla/ego_vehicle/depth_front/image`
- `/carla/ego_vehicle/depth_front/camera_info`
- `/carla/ego_vehicle/semantic_segmentation_front/image`
- `/carla/ego_vehicle/semantic_segmentation_front/camera_info`
- `/carla/ego_vehicle/semantic_lidar/point_cloud`
