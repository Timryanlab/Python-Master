# This gui is to unify the stage controls with the arduino stimulation controls
import tkinter as tk
import ardcom as ac
import serial
import pyasi
import csv

class myGui:
    def __init__(self, master, ser_stage, ser_arduino, bugable):
        self.root = master
        self.root.title("Stim-Stage Gui")
        self.armed = 0
        self.switch = 0
        self.debug_state = bugable
        # Run stage and stimulation setups
        self.setup_stage(ser_stage)

        self.setup_arduino(ser_arduino)
        #Definie a quit button
        self.close = tk.Button(self.root, text = "QUIT", command = self.shut_down, bg = "red")
        self.arduino.grid(row = 1, column = 0)
        self.stage.grid(row = 1, column = 1)
        self.close.grid(row = 0)

    def setup_stage(self, ser):
        # Setup widgets and positions related to the 'stage' portion of the gui
        self.ser_s = serial.Serial(ser, 9600)
        self.ser_s.write(b'vb z=3\r')
        self.ser_s.readline()
        self.stage = tk.Frame(self.root, bd = 2, relief = 'sunken') # Define frame for
        self.build_axes()
        self.build_focus()
        self.build_stage_marks()
        if self.debug_state >= 1:
            self.build_com_box()
            self.build_storm_wave()
            
        self.get_position()
        # Widget Positioning
        self.axes_frame.grid(row = 0, column = 0, padx = 2, pady = 2)
        self.focus_frame.grid(row = 0, column = 1, padx = 2, pady = 2)
        self.stage_mark_frame.grid(row = 1, column = 0, columnspan = 2, padx = 2, pady = 2)
        if self.debug_state >= 1:
            self.com_frame.grid(row = 2, column = 1, columnspan = 1, padx = 2, pady = 2)
            self.storm_wave_frame.grid(row = 2, column = 0)
    def build_axes(self):
        # Axes variables and current position widget
        self.axes_frame = tk.Frame(self.stage, bd = 2, relief = 'groove')
        self.axes_frame_title = tk.Label(self.axes_frame, text = "Axes Coordinates")
        self.x = tk.StringVar()
        self.y = tk.StringVar()
        self.z = tk.StringVar()
        self.x.set('0')
        self.y.set('0')
        self.z.set('0')
        self.current_x = tk.Label(self.axes_frame, textvariable = self.x)
        self.current_y = tk.Label(self.axes_frame, textvariable = self.y)
        self.current_z = tk.Label(self.axes_frame, textvariable = self.z)
        self.x_label = tk.Label(self.axes_frame, text = "X =")
        self.y_label = tk.Label(self.axes_frame, text = "Y =")
        self.z_label = tk.Label(self.axes_frame, text = "Z =")
        # Axes positioning
        self.axes_frame_title.grid(row = 0, column = 0, columnspan = 2)

        # Local Position Indicators
        pos_row = 1 
        pos_col = 0
        self.x_label.grid(row = pos_row, column = pos_col, sticky = 'E')
        self.y_label.grid(row = pos_row + 1, column = pos_col, sticky = 'E')
        self.z_label.grid(row = pos_row + 2, column = pos_col, sticky = 'E')

        self.current_x.grid(row = pos_row, column = pos_col + 1, sticky = 'W')
        self.current_y.grid(row = pos_row + 1, column = pos_col + 1, sticky = 'W')
        self.current_z.grid(row = pos_row + 2, column = pos_col + 1, sticky = 'W')

    def build_focus(self):
        # Focus Adjustment widget
        self.focus_frame = tk.Frame(self.stage, bd = 2, relief = 'groove')
        self.focus_name = tk.Label(self.focus_frame, text = 'Focus Adjustment')
        self.go_up = tk.Button(self.focus_frame, text = 'Shift Up', command = self.up_focus)
        self.go_down = tk.Button(self.focus_frame, text = 'Shift Down', command = self.dwn_focus)
        self.focus_shift = tk.StringVar()
        self.focus_shift.set(5)
        self.focus_amount = tk.Entry(self.focus_frame, textvariable = self.focus_shift, width =5)
        self.z_name = tk.Label(self.focus_frame, text = 'dz = ')
        # Positioning
        self.focus_name.grid(row = 0)
        self.go_up.grid(row=1, column = 0)
        self.go_down.grid(row = 2, column = 0)
        self. z_name.grid(row= 1, column = 1, sticky = 'E')
        self.focus_amount.grid(row = 1, column = 2, sticky = 'W')

    def build_stage_marks(self):
        self.stage_mark_frame = tk.Frame(self.stage, bd = 2, relief = 'groove')
        self.stage_mark_title = tk.Label(self.stage_mark_frame, text = 'Stage Marking')
        self.last_pos = tk.StringVar()
        self.last_pos.set(0)
        self.marked_pos = tk.Entry(self.stage_mark_frame, textvariable = self.last_pos, width = 5)
        self.marked_name = tk.Label(self.stage_mark_frame, text = 'Marked Position:')
        self.marks = [[0, 0 ,0]]
        self.mark_num = 0
        self.mark_text = tk.StringVar()
        self.mark_text.set('Out of 0 marks')
        self.tot_marks = tk.Label(self.stage_mark_frame, textvariable = self.mark_text)
        #Buttons
        self.mark = tk.Button(self.stage_mark_frame,text = 'Mark', command = self.new_mark_position, bg = 'blue',fg='yellow')
        self.up_mark = tk.Button(self.stage_mark_frame, text = 'Update mark', command = self.update_mark, bg = 'yellow')
        self.goto = tk.Button(self.stage_mark_frame, text = 'Go to mark', command = self.go_to_mark, bg = 'green')
        self.clear_marks = tk.Button(self.stage_mark_frame, text = 'Clear current mark', command = self.clear_mark, bg = 'red')
        self.load_last_marks = tk.Button(self.stage_mark_frame, text = 'Load Last Marks', command = self.load_marks)
        # Relative Positioning
        self.stage_mark_title.grid(row = 0, column = 1)        
        self.marked_name.grid(row = 1, column = 0)
        self.marked_pos.grid(row = 1, column = 1)
        self.tot_marks.grid(row = 1, column = 2)
        self.goto.grid(row = 3, column = 1)
        self.mark.grid(row = 3, column = 0)
        self.clear_marks.grid(row = 3, column = 2)
        self.up_mark.grid (row = 0, column = 0)
        self.load_last_marks.grid(row = 0, column = 2)

    def build_com_box(self):
        # Communication Box
        self.com_frame = tk.Frame(self.stage, bd = 2, relief = 'sunken')
        self.com_frame_title = tk.Label(self.com_frame, text = 'Stage Communication')
        self.cmd = tk.StringVar()
        self.cmd.set('A')
        self.command_line = tk.Entry(self.com_frame, textvariable = self.cmd)
        self.response = tk.StringVar()
        self.response.set('NA')
        self.command_response = tk.Label(self.com_frame, textvariable = self.response)
        self.cmd_button = tk.Button(self.com_frame, text = "Send", command = self.send_command)
        self.com_frame_title.grid(row = 0, column = 0 , columnspan = 2)
        self.command_line.grid(row = 1)
        self.command_response.grid(row = 2)
        self.cmd_button.grid(row = 1, column = 1)

    def build_storm_wave(self):
        self.storm_wave_frame = tk.Frame(self.stage, bd = 2, relief = 'sunken')
        self.wave_title = tk.Label(self.storm_wave_frame, text = 'Storm Wave')
        self.wave_dz = tk.StringVar()
        self.wave_range = tk.StringVar()
        self.wave_dz.set('10')
        self.wave_range.set('1')
        self.uv_state = 0
        self.wave_dz_title = tk.Label(self.storm_wave_frame, text = 'dz')
        self.wave_dz_entry = tk.Entry(self.storm_wave_frame, textvariable = self.wave_dz, width = 5)
        self.wave_range_title = tk.Label(self.storm_wave_frame, text = 'Range')
        self.wave_range_entry = tk.Entry(self.storm_wave_frame, textvariable = self.wave_range, width = 5)
        self.wave_dz_label = tk.Label(self.storm_wave_frame, text = 'nm')
        self.wave_range_label = tk.Label(self.storm_wave_frame, text = 'um')
        self.wave_button = tk.Button(self.storm_wave_frame, text = 'Turn Wave On', bg = 'magenta', command = self.wave)
        self.wave_state = 0
        self.wave_title.grid(row=0, column = 0)
        self.uv_laser_button = tk.Button(self.storm_wave_frame, text = "Turn 405 On", bg = 'magenta', command = self.uv_laser)
        self.wave_dz_title.grid(row=1, column = 0, sticky = 'E')
        self.wave_dz_entry.grid(row=1, column = 1, sticky = 'W')
        self.wave_dz_label.grid(row=1, column = 2)
        self.wave_range_title.grid(row=2, column = 0, sticky = 'E')
        self.wave_range_entry.grid(row=2, column = 1, sticky = 'W')
        self.wave_range_label.grid(row=2, column = 2)
        self.wave_button.grid(row = 3, column = 1)
        self.uv_laser_button.grid(row=3, column = 0)
        
    def setup_arduino(self, ser):
        self.ser_arduino = serial.Serial(ser, 9600)
        self.arduino = tk.Frame(self.root, bd =2, relief = 'groove')
        self.camera_frame_setup()

    def camera_frame_setup(self):
        w = 5
        # Define Com Response
        self.com_response = tk.StringVar()
        self.com_response.set("No Response")
        self.response = tk.Label(self.arduino, textvariable = self.com_response)
        self.response_text = tk.Label(self.arduino, text = "Arduino Response:")
        #Camera Frame
        self.camera = tk.StringVar()
        self.camera.set("0")
        self.camera_label = tk.Label(self.arduino, textvariable = self.camera)
        self.camera_text = tk.Label(self.arduino, text = 'Camera Frame')
        # Stimulus Section
        self.stimset = tk.StringVar() # define string variable for text
        self.stimset.set("10") # initialize string variable
        self.stimset.trace_add("write", self.change_stim)
        self.stimframe = tk.Entry(self.arduino, text = self.stimset, width = w)   # build entry 'widget' with text = string variable
        self.stimf_text = tk.Label(self.arduino, text = "Stimulate on Frame:") # Create text that goes with entry widget
        # Number of Stimuli / Train section
        self.stimN = tk.StringVar()
        self.stimN.set("1")
        self.stimN.trace_add("write" , self.change_stim_number)
        self.stimN_frame = tk.Entry(self.arduino, textvariable = self.stimN, width = w)   
        self.stimN_text = tk.Label(self.arduino, text = "Number of Stimuli/Train:")
        # Frequency Section
        self.frequency = tk.StringVar()
        self.frequency.set("1")
        self.frequency.trace_add("write", self.change_frequency)
        self.frequency_frame = tk.Entry(self.arduino, textvariable = self.frequency, width = w)  
        self.frequency_text = tk.Label(self.arduino, text = "Frequency of train:")
        self.hz = tk.Label(self.arduino, text = "Hz") 
        # Additional Parameters and startup
        self.com_response.set(ac.handshake(self.ser_arduino))
        if self.ser_arduino.inWaiting() > 0:
            self.ser_arduino.readline()
        self.com_label = tk.Label(self.arduino, textvariable = self.com_response)
        self.stim = tk.Button(self.arduino, text = "Stimulate!", command = self.stimulate)
        self.reset = tk.Button(self.arduino, text = "Reset Frames", command = self.reset_frame)
        self.toggle_switch = tk.Button(self.arduino, text = "Switcher", command = self.switcher, bg ='magenta')
        self.arm_button = tk.Button(self.arduino, text = 'Arm', command = self.arm, bg = 'magenta')
        #Start checking camera loop
        self.check_camera()
        
        ## Positioning ##

        # Stimulus Positioning
        stimrow = 0
        self.stimf_text.grid(row = stimrow,column = 0, sticky = 'E')
        self.stimframe.grid(row = stimrow, column = 1, sticky = 'W')
        self.camera_text.grid(row = stimrow, column = 2, sticky = 'E')
        self.camera_label.grid(row = stimrow + 1, column = 2, sticky = 'E')
        # Frequency Positioning
        freqrow = 2
        self.frequency_text.grid(row = freqrow, column = 0, sticky = 'E')
        self.frequency_frame.grid(row =freqrow, column = 1, sticky = 'W')
        self.hz.grid(row = freqrow, column = 2, sticky = 'W')

        # Number of Stimuli/train Positioning
        stimNrow = 1
        self.stimN_text.grid(row = stimNrow, column = 0, sticky = 'E')
        self.stimN_frame.grid(row = stimNrow, column = 1, sticky = 'W')
        # Arduino Response
        self.response.grid(row = freqrow + 1, column = 2, columnspan = 2)
        self.response_text.grid(row= freqrow +1, column =0, columnspan = 2)

        # Non-Entry Commands
        endrow = freqrow + 2
        self.stim.grid(row = endrow, column = 2, columnspan = 2)
        self.reset.grid(row = endrow, column = 0,columnspan = 2)
        self.arm_button.grid(row = endrow +1, column = 2, columnspan = 2)
        self.toggle_switch.grid(row = endrow + 1, column = 0,columnspan = 2)

    def check_camera(self):
        if(self.ser_arduino.in_waiting > 0):
            string = self.ser_arduino.readline()
            self.camera.set(string[0:-1])
        self.arduino.after(5,self.check_camera)

    def stimulate(self): 
        self.com_response.set(ac.send_command(self.ser_arduino,"S"))

    def switcher(self):
        self.com_response.set(ac.send_command(self.ser_arduino,"w"))
        self.switch = (self.switch + 1) % 2
        if self.switch is 1:
            self.toggle_switch.configure(bg='green')
        else:
            self.toggle_switch.configure(bg='magenta')

    def arm(self):
        self.com_response.set(ac.send_command(self.ser_arduino,"a"))
        self.armed = (self.armed + 1) % 2
        if self.armed is 1:
            self.arm_button.configure(bg='green')
        else:
            self.arm_button.configure(bg='magenta')

    def reset_frame(self):
        self.com_response.set(ac.send_command(self.ser_arduino,"r"))
        self.camera.set("0")

    # Text change Callback functions
    def wave(self):
        dz = float(self.wave_dz.get())
        um = float(self.wave_range.get())*1000
        steps = int(um/dz)
        self.wave_state = (self.wave_state + 1) % 2
        if self.wave_state is 1:
            pyasi.send_command(self.ser_s,'ZS X=' + str(dz/100) + ' Y=' + str(steps) + '\r')
            pyasi.send_command(self.ser_s,'TTL X=4\r')
            self.wave_button.configure(text = 'Turn Wave Off', bg = 'green')
        else:
            pyasi.send_command(self.ser_s,'TTL X=0\r')
            self.wave_button.configure(text = 'Turn Wave On', bg = 'magenta')

    def uv_laser(self):
        ac.send_command(self.ser_arduino, '4')
        self.uv_state = (self.uv_state + 1) % 2
        if self.uv_state is 1:
            self.uv_laser_button.config(text = 'Turn 405 Off', bg = 'green')
        else:
            self.uv_laser_button.config(text = 'Turn 405 On', bg = 'magenta')
    def change_stim(self, *args):
        self.com_response.set(ac.send_command(self.ser_arduino,"s",self.stimset.get()))

    def change_stim_number(self, *args):
        self.com_response.set(ac.send_command(self.ser_arduino,"n",self.stimN.get()))
    
    def send_command(self):
        self.response.set(pyasi.send_command(self.ser,self.cmd.get() + '\r'))

    def change_frequency(self, *args):
        self.com_response.set(ac.send_command(self.ser_arduino,"f",self.frequency.get()))

    def shut_down(self):
        # Close serial connections and destroy GUI
        self.ser_arduino.close()
        self.ser_s.close()
        self.root.destroy()

    def load_marks(self):
        f = open('positions.csv', 'r')
        reader = csv.reader(f)
        self.marks = list(reader)
        f.close()
        print(self.marks)
        self.mark_num = len(self.marks)
        self.mark_text.set('Out of {} marks'.format(self.mark_num))

    def up_focus(self):
        pyasi.send_command(self.ser_s,'r z={}\r'.format(self.focus_amount.get()))

    def dwn_focus(self):
        pyasi.send_command(self.ser_s,'r z=-{}\r'.format(self.focus_amount.get()))

    def go_to_mark(self):
        ind = int(self.last_pos.get())
        string = 'M X = {} Y = {} Z = {}\r'.format(self.marks[ind][0],self.marks[ind][1],self.marks[ind][2])
        pyasi.send_command(self.ser_s, string)

    def clear_mark(self):
        ind = int(self.last_pos.get())
        del self.marks[ind]
        if ind is self.mark_num: self.last_pos.set(int(self.last_pos.get()) - 1)
        self.mark_num = self.mark_num - 1
        self.mark_text.set('Out of {} marks'.format(self.mark_num))
        self.write_marks()

    def get_position(self):
        [x,y,z] = pyasi.get_positions(self.ser_s)
        self.x.set(x)
        self.y.set(y)
        self.z.set(z)
        self.stage.after(20, self.get_position)

    def new_mark_position(self):
        self.marks.append([self.x.get(),self.y.get(),self.z.get()])
        self.mark_num = self.mark_num + 1
        self.last_pos.set(self.mark_num)
        self.mark_text.set('Out of {} marks'.format(self.mark_num))
        self.write_marks()

    def update_mark(self):
        ind = int(self.last_pos.get())
        self.marks[ind] = [self.x.get(),self.y.get(),self.z.get()]

    def write_marks(self):
        f = open('positions.csv', 'w+')
        for i in range(len(self.marks)):
            f.write(str(self.marks[i][0]) + ',' + str(self.marks[i][1]) + ',' + str(self.marks[i][2])+'\n')
        f.close()

root = tk.Tk()
my = myGui(root,'COM2', 'COM6', 1)
root.mainloop()