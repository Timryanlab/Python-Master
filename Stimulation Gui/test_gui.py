import tkinter as tk
import ardcom as ac
import serial



    
class myGui:
    def __init__(self, master,com):
        self.ser = serial.Serial(com,9600)        
        self.root = master
        self.frame = tk.Frame(self.root)        
        self.com_response = tk.StringVar()
        self.com_response.set("No Response")
        w = 5 # Width of entry boxes

        # Response Section
        self.response = tk.Label(self.root, textvariable = self.com_response)
        self.response_text = tk.Label(self.root, text = "Arduino Response:")

        # Stimulus Section
        self.stimset = tk.StringVar() # define string variable for text
        self.stimset.set("10") # initialize string variable
        self.stimframe = tk.Entry(self.root, text = self.stimset, width = w)   # build entry 'widget' with text = string variable
        self.stimf_text = tk.Label(self.root, text = "Stimulate on Frame:") # Create text that goes with entry widget

        # Frequency Section
        self.frequency = tk.StringVar()
        self.frequency.set("1")
        self.frequency_frame = tk.Entry(self.root, text = self.frequency, width = w)  
        self.frequency_text = tk.Label(self.root, text = "Frequency of train:")
        self.hz = tk.Label(self.root, text = "Hz")  
        
        # Period Section
        self.period = tk.StringVar()
        self.period.set("1")
        self.period_frame = tk.Entry(self.root, text = self.period, width = w)   
        self.period_text = tk.Label(self.root, text = "Period of train:")
        self.s = tk.Label(self.root, text = "s")  
        
        # Number of Stimuli / Train section
        self.stimN = tk.StringVar()
        self.stimN.set("1")
        self.stimN_frame = tk.Entry(self.root, text = self.stimN, width = w)   
        self.stimN_text = tk.Label(self.root, text = "Number of Stimuli/Train:")

        # Pulse width section
        self.pwidth = tk.StringVar()
        self.pwidth.set("1")
        self.pwidth_frame = tk.Entry(self.root, text = self.pwidth, width = w)   
        self.pwidth_text = tk.Label(self.root, text = "Width of Pulse:")
        self.ms = tk.Label(self.root, text = "ms")  

        self.com_response.set(ac.handshake(self.ser))
        self.ser.readline()
        self.com_label = tk.Label(self.root, textvariable = self.com_response)
        self.close = tk.Button(self.root, text = "QUIT", command = self.shut_down, bg = "red")
        self.stim = tk.Button(self.root, text = "Stimulate!", command = self.stim)
        self.reset = tk.Button(self.root, text = "Reset Frames", command = self.reset_frame)
        self.toggle_switch = tk.Button(self.root, text = "Switcher", command = self.switcher)
        self.frame.grid()

        # Stimulus Positioning
        stimrow = 0
        self.stimf_text.grid(row = stimrow,column = 0, sticky = 'E')
        self.stimframe.grid(row = stimrow, column = 1, sticky = 'W')
        
        # Frequency Positioning
        freqrow = 2
        self.frequency_text.grid(row = freqrow, column = 0, sticky = 'E')
        self.frequency_frame.grid(row =freqrow, column = 1, sticky = 'W')
        self.hz.grid(row = freqrow, column = 2, sticky = 'W')

        # Period Positioning
        periodrow = 4
        self.period_text.grid(row = periodrow, column = 0, sticky = 'E')
        self.period_frame.grid(row = periodrow, column = 1 , sticky = 'W')
        self.s.grid(row = periodrow, column = 2, sticky = 'W')

        # Number of Stimuli/train Positioning
        stimNrow = 1
        self.stimN_text.grid(row = stimNrow, column = 0, sticky = 'E')
        self.stimN_frame.grid(row = stimNrow, column = 1, sticky = 'W')

        # Pulse width Positioning
        pwidthrow = 5
        self.pwidth_text.grid(row = pwidthrow, column = 0, sticky = 'E')
        self.pwidth_frame.grid(row = pwidthrow, column = 1, sticky = 'W')
        self.ms.grid(row = pwidthrow, column = 2, sticky = 'W')

        self.response.grid(row = pwidthrow + 1, column = 2, columnspan = 2)
        self.response_text.grid(row= pwidthrow +1, column =0, columnspan = 2)
        
        # Non-Entry Commands
        endrow = pwidthrow + 2
        self.stim.grid(row = endrow, column = 2, columnspan = 2)
        self.reset.grid(row = endrow, column = 0,columnspan = 2)
        self.toggle_switch.grid(row = endrow + 1, column = 0,columnspan = 2)
        self.close.grid(row = endrow + 1, column = 2, columnspan = 2)

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