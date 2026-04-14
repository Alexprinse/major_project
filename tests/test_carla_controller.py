import unittest

from drowsiness_detector import carla_controller
from drowsiness_detector.carla_controller import CarlaSafetyController


class _FakeVehicle:
    def __init__(self, location) -> None:
        self._location = location
        self.applied_controls = []

    def get_location(self):
        return self._location

    def apply_control(self, control) -> None:
        self.applied_controls.append(control)


@unittest.skipIf(carla_controller.carla is None, "CARLA Python API unavailable in test environment")
class CarlaSafetyControllerRouteArrivalTests(unittest.TestCase):
    def test_route_arrival_applies_stop_control(self) -> None:
        carla = carla_controller.carla
        controller = CarlaSafetyController()

        destination = (10.0, 20.0, 0.0)
        vehicle = _FakeVehicle(carla.Location(*destination))

        controller.vehicle = vehicle
        controller._route_destination = destination
        controller._route_active = True
        controller._route_agent = object()
        controller._route_parking_points = []

        status = controller.run_route_step()

        self.assertEqual(status, "arrived")
        self.assertFalse(controller._route_active)
        self.assertIsNone(controller._route_agent)
        self.assertGreater(len(vehicle.applied_controls), 0)

        stop_control = vehicle.applied_controls[-1]
        self.assertEqual(float(stop_control.throttle), 0.0)
        self.assertGreater(float(stop_control.brake), 0.0)


if __name__ == "__main__":
    unittest.main()
