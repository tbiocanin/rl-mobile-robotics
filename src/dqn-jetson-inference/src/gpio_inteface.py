#!/usr/bin/env python3

"""
Interface with the GPIO for movement commands
"""

import Jetson.GPIO as gpio
# import rospy

output_pins = {
    'JETSON_XAVIER': 18,
    'JETSON_NANO': 33,
    'JETSON_NX': 33,
    'CLARA_AGX_XAVIER': 18,
    'JETSON_TX2_NX': 32,
    'JETSON_ORIN': 18,
    'JETSON_ORIN_NX': 33,
    'JETSON_ORIN_NANO': 33
}
output_pin = output_pins.get(gpio.model, None)
if output_pin is None:
    raise Exception('PWM not supported on this board')

def setup(duty_cycle_value : int):
    """
    Setup pin states, pwm setup.
    """

    # left caterpillar setup
    gpio.setmode(gpio.BOARD)
    gpio.setup(32, gpio.OUT, initial=gpio.HIGH)
    pwm_left = gpio.PWM(32, 1000)
    gpio.setup(13, gpio.OUT)
    gpio.setup(12, gpio.OUT)

    # right caterpillar setup
    gpio.setup(33, gpio.OUT, initial=gpio.HIGH)
    pwm_right = gpio.PWM(33, 1000)
    gpio.setup(37, gpio.OUT)
    gpio.setup(40, gpio.OUT)

    pwm_left.start(duty_cycle_percent=duty_cycle_value)

    return [pwm_left, pwm_right]

def move_forward():
    
    # set the left caterpillar to go forward
    gpio.output(13, gpio.LOW)
    gpio.output(12, gpio.HIGH)

    # set the right caterpillar to go forward
    gpio.output(37, gpio.LOW)
    gpio.output(40, gpio.HIGH)

def move_left():
    # set the left caterpillar to go backward
    gpio.output(13, gpio.HIGH)
    gpio.output(12, gpio.LOW)

    # set the right caterpillar to go backward
    gpio.output(37, gpio.LOW)
    gpio.output(40, gpio.HIGH)

def move_right():
    # set the left caterpillar to go forward
    gpio.output(13, gpio.LOW)
    gpio.output(12, gpio.HIGH)

    # set the right caterpillar to go backward
    gpio.output(37, gpio.HIGH)
    gpio.output(40, gpio.LOW)

def stop():
    # set all pins to low to stop the movement
    gpio.output(13, gpio.LOW)
    gpio.output(12, gpio.LOW)

    # set the right caterpillar to go backward
    gpio.output(37, gpio.LOW)
    gpio.output(40, gpio.LOW)

def handle_user_prompt(user_in : str) -> None:

    if user_in == "0":
        move_forward()
    elif user_in == "1":
        move_left()
    elif user_in == "2":
        move_right()
    else:
        print("Invalid input!")

def main():
    [p_left, p_right] = setup(25)

    print("PWM running. Press CTRL+C to exit.")
    try:
        while True:
            user_in = input("Enter the next command: ")
            handle_user_prompt(user_in=user_in)
    except KeyboardInterrupt:

        # stop all pins and reset state
        stop()
        p_left.stop()
        p_right.stop()
        gpio.cleanup()

if __name__ == '__main__':
    main()