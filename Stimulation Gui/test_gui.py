import tkinter as tk
import ardcom as ac
import serial



    
class myGui:
    def __init__(self, master,com):
        self.ser = serial.Serial(com,9600)        
        self.root = master
        self.frame = tk.Frame(self.root)        
        self.com_response = tk.StringVar()
        self.stimset = tk.StringVar()
        self.stimset.set("10")
        self.stimframe = tk.Entry(self.root, text = self.stimset)        
        self.com_response.set(ac.handshake(self.ser))
        self.ser.readline()
        self.label = tk.Label(self.root, textvariable = self.com_response)
        self.close = tk.Button(self.root, text = "QUIT", command = self.shut_down)
        self.stim = tk.Button(self.root, text = "Stimulate!", command = self.stim)
        self.reset = tk.Button(self.root, text = "Reset Frames", command = self.reset_frame)
        self.toggle_switch = tk.Button(self.root, text = "Switcher", command = self.switcher)
        self.frame.grid()
        self.stim.grid(row = 2, column = 0)
        self.reset.grid(row = 2, column = 1)
        self.toggle_switch.grid(row = 2, column = 2)
        self.stimframe.grid(row = 0, column = 1)
        self.label.grid()
        self.close.grid()

    def stim(self):
        self.com_response.set(ac.send_command(self.ser,"S"))
    def switcher(self):
        self.com_response.set(ac.send_command(self.ser,"w"))
    def reset_frame(self):
        self.com_response.set(ac.send_command(self.ser,"r"))
    def shut_down(self):
        self.ser.close()
        self.root.destroy()

root = tk.Tk()
my = myGui(root,'COM4')
root.mainloop()