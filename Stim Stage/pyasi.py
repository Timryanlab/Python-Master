import serial

def get_positions(ser):
    ser.reset_input_buffer()
    ser.write(b'W X\r')
    string = ser.readline()
    try:
        x = float(string[3:-3])
    except:
        x = 'Error'
    
    ser.write(b'W Y\r')
    string = ser.readline()
    y = float(string[3:-3])

    ser.write(b'W Z\r')
    string = ser.readline()
    z = float(string[3:-3])

    return x, y, z

def send_command(ser,CMD):
    ser.write(CMD.encode())
    string = ser.readline()
    return string[0:-1]