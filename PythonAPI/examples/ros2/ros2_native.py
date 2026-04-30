#!/usr/bin/env python

# Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

import argparse
import json
import logging

import carla


def _normalize_sensor_config(sensor):
    if not isinstance(sensor, dict):
        return sensor

    attributes = dict(sensor.get("attributes", {}))
    reserved_keys = {"type", "id", "spawn_point", "attached_objects", "attributes"}
    for key, value in sensor.items():
        if key not in reserved_keys:
            attributes[key] = value

    normalized = dict(sensor)
    normalized["attributes"] = attributes
    return normalized


def _normalize_config(config):
    if "type" in config:
        normalized = dict(config)
        normalized["sensors"] = [_normalize_sensor_config(sensor) for sensor in config.get("sensors", [])]
        return normalized

    for obj in config.get("objects", []):
        if isinstance(obj, dict) and str(obj.get("type", "")).startswith("vehicle."):
            normalized = dict(obj)
            normalized["sensors"] = [_normalize_sensor_config(sensor) for sensor in obj.get("sensors", [])]
            return normalized

    raise ValueError("Configuration must define a vehicle 'type' or contain a vehicle entry in 'objects'")


def _set_bp_attribute_if_supported(bp, key, value):
    if bp.has_attribute(key):
        if isinstance(value, bool):
            serialized = "true" if value else "false"
        else:
            serialized = str(value)
        bp.set_attribute(str(key), serialized)


def _as_bool(value, default):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _setup_vehicle(world, config):
    logging.debug("Spawning vehicle: {}".format(config.get("type")))

    bp_library = world.get_blueprint_library()
    map_ = world.get_map()

    bp = bp_library.filter(config.get("type"))[0]
    vehicle_id = config.get("id")
    vehicle_tf_enabled = _as_bool(config.get("vehicle_ros_publish_tf"), False)

    _set_bp_attribute_if_supported(bp, "role_name", vehicle_id)
    _set_bp_attribute_if_supported(bp, "ros_name", vehicle_id)
    _set_bp_attribute_if_supported(bp, "ros_publish_tf", vehicle_tf_enabled)
    _set_bp_attribute_if_supported(bp, "ros_frame_id", config.get("vehicle_ros_frame_id", "map"))

    actor = None
    for spawn_point in map_.get_spawn_points():
        actor = world.try_spawn_actor(bp, spawn_point)
        if actor is not None:
            break

    if actor is None:
        raise RuntimeError("Failed to spawn vehicle at any map spawn point")

    logging.debug(
        "Vehicle ROS attrs: %s",
        {k: v for k, v in actor.attributes.items() if "ros" in k or "frame" in k or "role" in k}
    )

    return actor


def _setup_sensors(world, vehicle, sensors_config, default_sensor_frame_id):
    bp_library = world.get_blueprint_library()

    sensors = []
    for sensor in sensors_config:
        logging.debug("Spawning sensor: {}".format(sensor))
        sensor_attributes = sensor.get("attributes", {})

        matches = bp_library.filter(sensor.get("type"))
        if not matches:
            logging.warning(
                "Sensor '%s' (%s) is not available in the CARLA blueprint library; skipping",
                sensor.get("id"),
                sensor.get("type")
            )
            continue

        bp = matches[0]
        _set_bp_attribute_if_supported(bp, "ros_name", sensor.get("id"))
        _set_bp_attribute_if_supported(bp, "role_name", sensor.get("id"))

        # If a sensor does not specify ros_frame_id, use a stable default frame.
        if "ros_frame_id" not in sensor_attributes:
            _set_bp_attribute_if_supported(bp, "ros_frame_id", default_sensor_frame_id)
        if "ros_publish_tf" not in sensor_attributes:
            _set_bp_attribute_if_supported(bp, "ros_publish_tf", False)

        for key, value in sensor_attributes.items():
            bp.set_attribute(str(key), str(value))

        spawn_point = sensor.get("spawn_point", {})
        wp = carla.Transform(
            location=carla.Location(
                x=spawn_point.get("x", 0.0),
                y=-spawn_point.get("y", 0.0),
                z=spawn_point.get("z", 0.0),
            ),
            rotation=carla.Rotation(
                roll=spawn_point.get("roll", 0.0),
                pitch=-spawn_point.get("pitch", 0.0),
                yaw=-spawn_point.get("yaw", 0.0),
            )
        )

        sensors.append(
            world.spawn_actor(
                bp,
                wp,
                attach_to=vehicle
            )
        )

        logging.debug(
            "Sensor '%s' ROS attrs: %s",
            sensor.get("id"),
            {k: v for k, v in sensors[-1].attributes.items() if "ros" in k or "frame" in k or "role" in k}
        )

        # Not every actor class exposes ROS native publishing (for example lane invasion).
        if hasattr(sensors[-1], "enable_for_ros"):
            sensors[-1].enable_for_ros()
        else:
            logging.warning(
                "Sensor '%s' (%s) does not support enable_for_ros(); skipping ROS bridge for this sensor class",
                sensor.get("id"),
                sensor.get("type")
            )

    return sensors


def main(args):

    world = None
    vehicle = None
    sensors = []
    original_settings = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.get_world()

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        with open(args.file) as f:
            config = json.load(f)

        config = _normalize_config(config)

        vehicle = _setup_vehicle(world, config)
        vehicle_frame_id = config.get("id", "ego_vehicle")
        vehicle_tf_enabled = _as_bool(config.get("vehicle_ros_publish_tf"), False)
        default_sensor_frame_id = config.get(
            "sensor_ros_frame_id",
            vehicle_frame_id if vehicle_tf_enabled else config.get("vehicle_ros_frame_id", "map")
        )
        sensors = _setup_sensors(world, vehicle, config.get("sensors", []), default_sensor_frame_id)

        _ = world.tick()

        vehicle.set_autopilot(True)

        logging.info("Running...")

        while True:
            _ = world.tick()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

    finally:
        if original_settings:
            world.apply_settings(original_settings)

        for sensor in sensors:
            sensor.destroy()

        if vehicle:
            vehicle.destroy()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='CARLA ROS2 native')
    argparser.add_argument('--host', metavar='H', default='localhost', help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000, type=int, help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-f', '--file', default='', required=True, help='File to be executed')
    argparser.add_argument('-v', '--verbose', action='store_true', dest='debug', help='print debug information')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('Listening to server %s:%s', args.host, args.port)

    main(args)
