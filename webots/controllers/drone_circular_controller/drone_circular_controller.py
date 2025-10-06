from controller import Supervisor
import math
import sys


def parse_arguments(arguments):
    radius = 3.5
    height = 3.0
    period = 25.0
    center_def = "SPOT"
    for arg in arguments:
        if arg.startswith("--radius="):
            radius = float(arg.split("=", 1)[1])
        elif arg.startswith("--height="):
            height = float(arg.split("=", 1)[1])
        elif arg.startswith("--period="):
            period = float(arg.split("=", 1)[1])
        elif arg.startswith("--center-def="):
            center_def = arg.split("=", 1)[1]
    return radius, height, period, center_def


def main():
    supervisor = Supervisor()
    time_step = int(supervisor.getBasicTimeStep())
    arguments = sys.argv[1:]
    radius, height, period, center_def = parse_arguments(arguments)

    drone_node = supervisor.getSelf()
    translation_field = drone_node.getField("translation")
    rotation_field = drone_node.getField("rotation")

    center_node = supervisor.getFromDef(center_def)
    start_time = supervisor.getTime()

    while supervisor.step(time_step) != -1:
        elapsed = supervisor.getTime() - start_time
        angle = (elapsed / period) * 2.0 * math.pi if period > 0 else 0.0

        if center_node is not None:
            center_position = list(center_node.getPosition())
        else:
            center_position = [0.0, 0.0, 0.0]

        target_x = center_position[0] + radius * math.cos(angle)
        target_y = center_position[1] + radius * math.sin(angle)
        target_z = center_position[2] + height

        translation_field.setSFVec3f([target_x, target_y, target_z])

        heading = angle + math.pi / 2.0
        rotation_field.setSFRotation([0.0, 0.0, 1.0, heading])


if __name__ == "__main__":
    main()
