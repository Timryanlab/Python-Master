import serial
import time

# Short library for arduino stimulation commands, this will allow the user to input a command and return the result

def handshake(ser): # Clear initial buffer
    string = ser.readline()
    return string[0:-1]

def send_command(ser,cmd,var = None): # Switch Statement
    switch = {
        "r": reset_count,
        "S": run_train,
        "s": set_stim,
        "w": toggle_switcher,
        "f": set_frequency,
        "n": set_stim_number,
        "P": set_pulse_width,
        "p": set_period,
        'a': arm,
        "R": set_repeat, #--KB 
        "4": uv_laser,
        "t": toggle_camera,
    }
    func = switch.get(cmd,send_string)
    if(var != None): 
        func(ser,var)
    else:
        func(ser)


    string = ser.readline()
    return string[0:-1]
def arm(ser):
    ser.write(b'a')
def send_string(ser,string):
    ser.write(string.encode())

def reset_count(ser):
    ser.write(b"r")

def toggle_camera(ser):
    ser.write(b"t")

def run_train(ser):
    ser.write(b"S")

def set_stim(ser,stimulus):
    string = "s" + str(stimulus)
    send_string(ser,string)

def toggle_switcher(ser):
    ser.write(b"w")

def set_frequency(ser, f):
    string = "f" + str(f)
    send_string(ser,string)

def set_stim_number(ser,N):
    string = "N" + str(N)
    send_string(ser,string)

def set_pulse_width(ser,N):
    string = "P" + str(N)
    send_string(ser,string)

def set_period(ser,N):
    string = "p" + str(N)
    send_string(ser,string)

def set_repeat(ser,N): #for repeater--KB
    string = "R" + str(N)
    send_string(ser,string)

def uv_laser(ser):
    ser.write(b'4')

switch = {
        "r": reset_count,
        "S": run_train,
        "s": set_stim,
        "w": toggle_switcher,
        "f": set_frequency,
        "n": set_stim_number,
        "P": set_pulse_width,
        "p": set_period,
        'a': arm,
        "R": set_repeat, #--KB 
        "4": uv_laser,
        "t": toggle_camera,
    }
print(switch)
