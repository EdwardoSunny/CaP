"""Home the xArm robot to its default position with gripper open."""

from ril_env.xarm_controller import XArm, XArmConfig


def main():
    xarm_config = XArmConfig()

    with XArm(xarm_config) as arm:
        try:
            print("Homing robot...")
            arm.home()
            print("Robot homed successfully.")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
