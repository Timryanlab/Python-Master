import serial
import numpy as np

while True:
    try:
        ser = serial.Serial('COM3',9600)
        if ser.isOpen:
            print('Connected, measuring data...')
            break
    except:
        pass
count = 0
measures = 1000
x = np.zeros(measures)
ser.reset_input_buffer()
while count < measures:
    if ser.in_waiting > 0:
        try:
            out = ser.readline()
            x[count] = float(out[:-1])
            count += 1
        except:
            ser.reset_input_buffer()
ser.close()
print("Signal to Noise Ratio is:" + str(np.power(np.mean(x)/np.std(x),2)))
print("Average Current :" + str( np.mean(x)) + ' mA')
print("Signal variance:" + str( np.power(np.std(x),2)))
print("Measured " + str(measures)+ " times")