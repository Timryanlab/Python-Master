import serial
import time

def handshake(ser): # Clear initial buffer
    ser.readline()

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
    }
    func = switch.get(cmd,lambda:'No Dice')
    if(var != None): 
        func(ser,var)
    else:
        func(ser)
    string = ser.readline()
    print(string[0:-1])

def send_string(ser,string):
    ser.write(string.encode())

def reset_count(ser):
    ser.write(b"r")

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


ser = serial.Serial('COM4', 9600)
handshake(ser)
if(ser.in_waiting > 0): ser.readline()

send_command(ser,'s',12)
print("After Stimulus Set")
send_command(ser,'p',34)
print("After frequency Set")
send_command(ser,'w')
print("After switcher set")
send_command(ser,'S')


ser.close()