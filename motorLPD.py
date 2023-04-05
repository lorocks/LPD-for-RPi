import serial

arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
arduino.reset_input_buffer()

while True:
    num = input("Enter a number or q to exit: ")
    if num == 'q':
        break
    arduino.write(bytes(num, 'utf-8'))
    time.sleep(0.05)
    print("done")