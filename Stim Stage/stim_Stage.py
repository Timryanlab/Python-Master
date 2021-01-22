# This gui is to unify the stage controls with the arduino stimulation controls
# Stim_Stage 1.1
# This is the basic GUI with support for Andrew's scope
# There are place holders for scope specialization
import tkinter as tk
import ardcom as ac
import serial
import time
import pyasi
import csv
import os
from datetime import datetime

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
        if self.debug_state == 1: # Andrew's Scope's configuration
            self.build_com_box()
            self.build_storm_wave()
            self.com_frame.grid(row = 2, column = 1, columnspan = 2, padx = 2, pady = 2)
            self.storm_wave_frame.grid(row = 2, column = 0)
            self.build_channel_calibration()
            #self.build_dual_camera()
            #self.dual_camera_frame.grid(row = 2, column = 0)
            self.cal_frame.grid(row=3)
            self.zx=0
            self.zy=0
        if self.debug_state == 2: # SYN-ATP scope Config
            print('NO')
        if self.debug_state == 3: # The Bear scope Config
            self.build_dual_camera()
            self.dual_camera_frame.grid(row = 2, column = 0)
        if self.debug_state == 4: # Michelle scope Config
            print('NO')

    # Functions are listed alphabetically for ease of searching
    def arm(self):
        self.com_response.set(ac.send_command(self.ser_arduino,"a"))
        self.armed = (self.armed + 1) % 2
        if self.armed is 1:
            self.arm_button.configure(bg='green')
        else:
            self.arm_button.configure(bg='magenta')

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

    def build_channel_calibration(self):
        self.cal_frame = tk.Frame(self.stage, bd = 2, relief = 'sunken')
        self.channel_button = tk.Button(self.cal_frame, text = "Scan", command = self.channel_calibration)
        self.cal_range_z = tk.StringVar()    
        self.cal_range_z.set('4')
        self.cal_dz = tk.StringVar()    
        self.cal_dz.set('0.02')
        
        self.tdel = tk.StringVar()
        self.tdel.set('0.01')
        self.tex = tk.StringVar()
        self.tex.set('0.045')
        self.tdel_entry = tk.Entry(self.cal_frame, textvariable = self.tdel, width =5)
        self.tex_entry = tk.Entry(self.cal_frame, textvariable = self.tex, width =5)
        self.cal_z_entry = tk.Entry(self.cal_frame, textvariable= self.cal_range_z, width =5)
        self.frames_for_scan = tk.StringVar()
        self.frames_for_scan.set('1')
        self.frames_for_scan_entry = tk.Entry(self.cal_frame, textvariable = self.frames_for_scan, width =5)
        self.frames_for_scan_label = tk.Label(self.cal_frame,text = 'Frames per step')        
        self.cal_dz_entry = tk.Entry(self.cal_frame, textvariable= self.cal_dz, width =5)
        
        self.cal_title = tk.Label(self.cal_frame, text = 'Raster Scan')
        self.cal_z_label = tk.Label(self.cal_frame,text = ' z range in um')        
        self.cal_z_steps = tk.Label(self.cal_frame,text = ' z steps um')  
        self.tdel_unit = tk.Label(self.cal_frame, text = 's Delay')
        self.tex_unit = tk.Label(self.cal_frame, text = 's Exposure')
        self.dump_button = tk.Button(self.cal_frame, text ='DUMP', command = self.dump_stage)
        # Gridding
        self.cal_z_entry.grid(row=1, column = 0)
        self.cal_z_label.grid(row=1, column = 1)
        self.cal_dz_entry.grid(row = 1, column = 2)
        

        self.channel_button.grid(row=1, column = 4, rowspan = 2)

        self.tdel_entry.grid(row = 3, column = 0)
        self.tex_entry.grid(row = 4, column = 0)
        self.tdel_unit.grid(row = 3, column =1)
        self.tex_unit.grid(row = 4, column =1)
        self.frames_for_scan_entry.grid(row = 3, column = 2)
        self.frames_for_scan_label.grid(row = 3, column = 3)


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
        self.cmd_button = tk.Button(self.com_frame, text = "Send", command = self.send_asi_command)
        self.reset_button = tk.Button(self.com_frame, text = "RESET", command = self.stage_reset)
        
        self.com_frame_title.grid(row = 0, column = 0 , columnspan = 2)
        self.command_line.grid(row = 1)
        self.command_response.grid(row = 2)
        self.reset_button.grid(row = 2, column = 1)
        self.cmd_button.grid(row = 1, column = 1)
        
    def build_dual_camera(self):
        self.dual_camera_frame =tk.Frame(self.root, bd =2 , relief = 'sunken')
        self.external_trigger_button = tk.Button(self.dual_camera_frame, text = 'Ext. Trigger', bg = 'magenta', command = self.external_trigger)
        self.inverter_button = tk.Button(self.dual_camera_frame, text = 'Inverter', bg = 'magenta', command = self.invert)
        self.simultaneous_button = tk.Button(self.dual_camera_frame, text = 'Simultaneous', bg = 'magenta', command = self.simultaneous)
        self.simul_tog = 0
        self.inver_tog = 0
        self.exttr_tog = 0
        self.camera_period_label = tk.Label(self.dual_camera_frame, text = 'Camera period')
        self.camera_period_units = tk.Label(self.dual_camera_frame, text = 'ms')
        self.camera_period = tk.StringVar()
        self.camera_period.set('100')
        self.camera_period.trace_add("write", self.change_period)
        self.camera_period_entry = tk.Entry(self.dual_camera_frame, textvariable = self.camera_period)
        #Organization
        self.external_trigger_button.grid(row = 1, column = 0, columnspan = 3, padx = 2, pady = 2)
        self.inverter_button.grid(row = 1, column = 3, columnspan = 3, padx = 2, pady = 2)
        self.simultaneous_button.grid(row = 2, column = 0, columnspan = 3, padx = 2, pady = 2)
        self.camera_period_label.grid(row = 2, column = 3, columnspan = 1, padx = 2, pady = 2)
        self.camera_period_entry.grid(row = 2, column = 4, columnspan = 1, padx = 2, pady = 2)
        self.camera_period_units.grid(row = 2, column = 5, columnspan = 1, padx = 2, pady = 2)

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

    def build_storm_wave(self):
        self.storm_wave_frame = tk.Frame(self.stage, bd = 2, relief = 'sunken')
        self.wave_title = tk.Label(self.storm_wave_frame, text = 'Storm Wave')
        self.wave_dz = tk.StringVar()
        self.wave_range = tk.StringVar()
        self.wave_dz.set('20')
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
 
    def channel_calibration(self):
        
        z_start = self.z.get() # Grab initial z position for after scan placement
        tdel = float(self.tdel.get())
        tex = float(self.tex.get())
        z_range = 10*float(self.cal_range_z.get())
        dz = 10*float(self.cal_dz.get())
        z_steps = int(z_range/dz)
        # Offset the stage by half
        stng = 'R Z=' + str(-z_range/2 - dz) + '\r'
        pyasi.send_command(self.ser_s,stng)

        frames = int(self.frames_for_scan.get()) # Define a number of frames to get images over
        
        time.sleep(tdel)
        for c in range(z_steps):
            stng = 'R Z='+ str(dz) +'\r'
            
            pyasi.send_command(self.ser_s,stng)
            time.sleep(tdel)
            
            for n in range(frames):
                ac.send_command(self.ser_arduino,'t')
                time.sleep(tex+tdel)
                

        #strng = 'R Z=' + str(-z_range/2) + '\r' $ Relative movement back to where z should be is slightly off, perhaps due to the dz component
        strng = 'M Z=' + str(z_start) + '\r' # Force movement back to original z
        pyasi.send_command(self.ser_s, strng)

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
        self.stimN.set("20")
        self.stimN.trace_add("write" , self.change_stim_number)
        self.stimN_frame = tk.Entry(self.arduino, textvariable = self.stimN, width = w)   
        self.stimN_text = tk.Label(self.arduino, text = "Number of Stimuli/Train:")
        # Frequency Section
        self.frequency = tk.StringVar()
        self.frequency.set("20")
        self.frequency.trace_add("write", self.change_frequency)
        self.frequency_frame = tk.Entry(self.arduino, textvariable = self.frequency, width = w)  
        self.frequency_text = tk.Label(self.arduino, text = "Frequency of train:")
        self.hz = tk.Label(self.arduino, text = "Hz") 
        # Repeater section--KB
        #self.repeat_tog = 0
        self.repeat_every = tk.StringVar() #define string variable for text
        self.repeat_every.set("0") #initialize string variable
        self.repeat_every.trace_add("write", self.change_repeats)
        self.repeatframe = tk.Entry(self.arduino, textvariable = self.repeat_every, width = w)   # build entry 'widget' with text = string variable
        self.repeatf_text_before = tk.Label(self.arduino, text = "Stimulate every:") # text that goes with entry widget
        self.repeatf_text_after = tk.Label(self.arduino, text = "frames") # text that goes with entry widget

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
        # Repeater Positioning
        repeatrow = 3 
        self.repeatf_text_before.grid(row = repeatrow, column = 0, sticky = 'E')
        self.repeatframe.grid(row = repeatrow, column = 1, sticky = 'W')
        self.repeatf_text_after.grid(row = repeatrow, column = 2, sticky = 'W')

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
   
    def change_frequency(self, *args):
        self.com_response.set(ac.send_command(self.ser_arduino,"f",self.frequency.get()))

    def change_period(self, *args):
        self.com_response.set(ac.send_command(self.ser_arduino,"c",self.camera_period.get()))
    
    def change_stim(self, *args):
        self.com_response.set(ac.send_command(self.ser_arduino,"s",self.stimset.get()))

    def change_stim_number(self, *args):
        self.com_response.set(ac.send_command(self.ser_arduino,"n",self.stimN.get()))
    
    def change_repeats(self, *args): # --KB
        self.com_response.set(ac.send_command(self.ser_arduino, "R", self.repeat_every.get()))
    
    def check_camera(self):
        if(self.ser_arduino.in_waiting > 0):
            string = self.ser_arduino.readline()
            self.camera.set(string[0:-1])
        self.arduino.after(1,self.check_camera) # 1 ms check keeps the counter up to par w/ faster imaging experiments

    def clear_arduino(self):
        self.ser_arduino.reset_input_buffer()
        self.ser_arduino.reset_output_buffer()

    def clear_both(self):
        self.clear_arduino
        self.clear_stage

    def clear_stage(self):
        self.ser_s.reset_input_buffer()
        self.ser_s.reset_output_buffer()

    def clear_mark(self):
        ind = int(self.last_pos.get())
        del self.marks[ind]
        if ind is self.mark_num: self.last_pos.set(int(self.last_pos.get()) - 1)
        self.mark_num = self.mark_num - 1
        self.mark_text.set('Out of {} marks'.format(self.mark_num))
        self.write_marks()

    def dump_stage(self):
        stng = pyasi.send_command(self.ser_s,'DU\r')
        f = open('dump.txt', 'w+')
        now = datetime.now()
        f.write(str(now) + ':\r'+ stng.decode())
        f.close()


    def dwn_focus(self):
        pyasi.send_command(self.ser_s,'r z=-{}\r'.format(self.focus_amount.get()))

    def external_trigger(self):
        self.com_response.set(ac.send_command(self.ser_arduino,"e"))
        self.exttr_tog = (self.exttr_tog + 1)%2
        if self.exttr_tog is 1: 
            self.external_trigger_button.configure(bg = 'green')
        else:
            self.external_trigger_button.configure(bg = 'magenta')

    def invert(self):
        self.com_response.set(ac.send_command(self.ser_arduino,"i"))
        self.inver_tog = (self.inver_tog + 1)%2
        if self.inver_tog is 1: 
            self.inverter_button.configure(bg = 'green')
        else:
            self.inverter_button.configure(bg = 'magenta')
 
    def get_position(self):
        fl = 0
        while fl is 0:
            try:
                [x,y,z] = pyasi.get_positions(self.ser_s)
                fl = 1
            except:
                fl = 0
        self.x.set(x)
        self.y.set(y)
        self.z.set(z)
        self.stage.after(20, self.get_position)

    def go_to_mark(self):
        ind = int(self.last_pos.get())
        string = 'M X = {} Y = {} Z = {}\r'.format(self.marks[ind][0],self.marks[ind][1],self.marks[ind][2])
        pyasi.send_command(self.ser_s, string)

    def load_marks(self):
        f = open('positions.csv', 'r')
        reader = csv.reader(f)
        self.marks = list(reader)
        f.close()
        print(self.marks)
        self.mark_num = len(self.marks)
        self.mark_text.set('Out of {} marks'.format(self.mark_num))

    def new_mark_position(self):
        self.marks.append([self.x.get(),self.y.get(),self.z.get()])
        self.mark_num = self.mark_num + 1
        self.last_pos.set(self.mark_num)
        self.mark_text.set('Out of {} marks'.format(self.mark_num))
        self.write_marks()

    def reset_frame(self):
        self.com_response.set(ac.send_command(self.ser_arduino,"r"))
        self.camera.set("0")

    def send_asi_command(self):
        self.response.set(pyasi.send_command(self.ser_s,self.cmd.get() + '\r'))
        #print(self.response.get())
      
    def setup_arduino(self, ser):
        flag = 1
        print('Connecting to Arduino... ', end='')
        while flag is 1:
            try:
                self.ser_arduino = serial.Serial(ser, 9600)
                flag = 0
            except:
                flag = 1
        print('Complete\n')
        self.arduino = tk.Frame(self.root, bd =2, relief = 'groove')
        self.camera_frame_setup()
 
    def setup_stage(self, ser):
        # Setup widgets and positions related to the 'stage' portion of the gui
        flag = 1
        print('Opening Stage communication... ', end ='')
        
        while flag is 1:
            try:
                self.ser_s = serial.Serial(ser, 9600)
                flag = 0
            except:
                flag = 1
        print('complete\n')
        self.stage_reset()
        self.stage = tk.Frame(self.root, bd = 2, relief = 'sunken') # Define frame for
        self.build_axes()
        self.build_focus()
        self.build_stage_marks()            
        self.get_position()
        # Widget Positioning
        self.axes_frame.grid(row = 0, column = 0, padx = 2, pady = 2)
        self.focus_frame.grid(row = 0, column = 1, padx = 2, pady = 2)
        self.stage_mark_frame.grid(row = 1, column = 0, columnspan = 2, padx = 2, pady = 2)
 
    def shut_down(self):
        # Close serial connections and destroy GUI
        self.ser_arduino.close()
        self.ser_s.close()
        self.root.destroy()

    def simultaneous(self):
        self.com_response.set(ac.send_command(self.ser_arduino,"k"))
        self.simul_tog = (self.simul_tog + 1)%2
        if self.simul_tog is 1: 
            self.simultaneous_button.configure(bg = 'green')
        else:
            self.simultaneous_button.configure(bg = 'magenta')

    def stage_reset(self):
        self.ser_s.write(b'reset\r')
        self.ser_s.readline()
        self.ser_s.write(b'vb z=3\r')
        self.ser_s.readline()

    def stimulate(self): 
        self.com_response.set(ac.send_command(self.ser_arduino,"S"))

    def switcher(self):
        self.com_response.set(ac.send_command(self.ser_arduino,"w"))
        self.switch = (self.switch + 1) % 2
        if self.switch is 1:
            self.toggle_switch.configure(bg='green')
        else:
            self.toggle_switch.configure(bg='magenta')

    def update_mark(self):
        ind = int(self.last_pos.get())
        self.marks[ind] = [self.x.get(),self.y.get(),self.z.get()]

    def up_focus(self):
        pyasi.send_command(self.ser_s,'r z={}\r'.format(self.focus_amount.get()))

    def uv_laser(self):
        ac.send_command(self.ser_arduino, '4')
        self.uv_state = (self.uv_state + 1) % 2
        if self.uv_state is 1:
            self.uv_laser_button.config(text = 'Turn 405 Off', bg = 'green')
        else:
            self.uv_laser_button.config(text = 'Turn 405 On', bg = 'magenta')

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
 
    def write_marks(self):
        f = open('positions.csv', 'w+')
        for i in range(len(self.marks)):
            f.write(str(self.marks[i][0]) + ',' + str(self.marks[i][1]) + ',' + str(self.marks[i][2])+'\n')
        f.close()

def get_scope_number():
    # Dictionary of scope numbers corresponding to scope login    
    scope_dict = {
        'Andrew' : 1, #super-res scope
        'JIMMY'  : 2, #syn-atp scope
        'The Bear':3, # 2color diff-limited scope
        'RyanLab':4 # Widefield casual scope
    }
    return scope_dict[os.getlogin()]

root = tk.Tk()
my = myGui(root,'COM2', 'COM6', get_scope_number())
root.mainloop()