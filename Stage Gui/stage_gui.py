import serial
import tkinter as tk
import csv
import pyasi

class stage_gui:
    def __init__(self, master, ser):
        self.root = master
        self.root.title("Stage Gui")
        self.frame = tk.Frame(self.root)
        self.close = tk.Button(self.root, text = "QUIT", command = self.shut_down, bg = "red")
        self.Start_Serial(ser)
        
        # Axes variables and current position
        self.axes_frame = tk.Frame(self.root, bd = 2, relief = 'groove')
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
        
        # Focus Adjustment
        self.focus_frame = tk.Frame(self.root, bd = 2, relief = 'groove')
        self.focus_name = tk.Label(self.focus_frame, text = 'Focus Adjustment')
        self.go_up = tk.Button(self.focus_frame, text = 'Shift Up', command = self.up_focus)
        self.go_down = tk.Button(self.focus_frame, text = 'Shift Down', command = self.dwn_focus)
        self.focus_shift = tk.StringVar()
        self.focus_shift.set(5)
        self.focus_amount = tk.Entry(self.focus_frame, textvariable = self.focus_shift, width =5)
        self.z_name = tk.Label(self.focus_frame, text = 'dz = ')
        self.focus_frame.grid(row = 1, column = 1)
        self.focus_name.grid(row = 0)
        self.go_up.grid(row=1, column = 0)
        self.go_down.grid(row = 2, column = 0)
        self. z_name.grid(row= 1, column = 1, sticky = 'E')
        self.focus_amount.grid(row = 1, column = 2, sticky = 'W')

        # Communication Box
        self.com_frame = tk.Frame(self.root, bd = 2, relief = 'sunken')
        self.com_frame_title = tk.Label(self.com_frame, text = 'Stage Communication')
        self.cmd = tk.StringVar()
        self.cmd.set('A')
        self.command_line = tk.Entry(self.com_frame, textvariable = self.cmd)
        self.response = tk.StringVar()
        self.response.set('NA')
        self.command_response = tk.Label(self.com_frame, textvariable = self.response)
        self.cmd_button = tk.Button(self.com_frame, text = "Send", command = self.send_command)

        # Stage Marking Region
        self.stage_mark_frame = tk.Frame(self.root, bd = 2, relief = 'groove')
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
        self.mark = tk.Button(self.stage_mark_frame,text = 'Mark', command = self.new_mark_position, bg = 'blue')
        self.up_mark = tk.Button(self.stage_mark_frame, text = 'Update mark', command = self.update_mark, bg = 'yellow')
        self.goto = tk.Button(self.stage_mark_frame, text = 'Go to mark', command = self.go_to_mark, bg = 'green')
        self.clear_marks = tk.Button(self.stage_mark_frame, text = 'Clear current mark', command = self.clear_mark, bg = 'red')
        self.load_last_marks = tk.Button(self.stage_mark_frame, text = 'Load Last Marks', command = self.load_marks)

        # Placement
        self.stage_mark_frame.grid(row = 0, column = 1)
        # Relative
        self.stage_mark_title.grid(row = 0, column = 1)        
        self.marked_name.grid(row = 1, column = 0)
        self.marked_pos.grid(row = 1, column = 1)
        self.tot_marks.grid(row = 1, column = 2)
        self.goto.grid(row = 3, column = 1)
        self.mark.grid(row = 3, column = 0)
        self.clear_marks.grid(row = 3, column = 2)
        self.up_mark.grid (row = 4, column = 0)
        self.load_last_marks.grid(row = 0, column = 2)

        #####################Widget Positioning######################

        # Axes positioning
        pos_col = 0
        pos_row = 0
        self.axes_frame.grid(row = pos_row, column = 0, padx = 2, pady = 2)
        self.axes_frame_title.grid(row = pos_row, column = 0, columnspan = 2)

        # Position Indicators
        pos_row = pos_row +1 
        self.x_label.grid(row = pos_row, column = pos_col, sticky = 'E')
        self.y_label.grid(row = pos_row + 1, column = pos_col, sticky = 'E')
        self.z_label.grid(row = pos_row + 2, column = pos_col, sticky = 'E')

        self.current_x.grid(row = pos_row, column = pos_col + 1, sticky = 'W')
        self.current_y.grid(row = pos_row + 1, column = pos_col + 1, sticky = 'W')
        self.current_z.grid(row = pos_row + 2, column = pos_col + 1, sticky = 'W')

        # Com Line
        com_row = 1 
        com_col = 0
        self.com_frame.grid(row = com_row, column = com_col, padx = 2, pady = 2)
        self.com_frame_title.grid(row = 0, column = 0 , columnspan = 2)
        self.command_line.grid(row = 1)
        self.command_response.grid(row = 2)
        self.cmd_button.grid(row = 1, column = 1)
        self.close.grid()
        self.get_position()

    def load_marks(self):
        f = open('positions.csv', 'r')
        reader = csv.reader(f)
        self.marks = list(reader)
        f.close()
        print(self.marks)
        self.mark_num = len(self.marks)
        self.mark_text.set('Out of {} marks'.format(self.mark_num))

    def up_focus(self):
        pyasi.send_command(self.ser,'r z={}\r'.format(self.focus_amount.get()))

    def dwn_focus(self):
        pyasi.send_command(self.ser,'r z=-{}\r'.format(self.focus_amount.get()))

    def go_to_mark(self):
        ind = int(self.last_pos.get())
        string = 'M X = {} Y = {} Z = {}\r'.format(self.marks[ind][0],self.marks[ind][1],self.marks[ind][2])
        pyasi.send_command(self.ser, string)

    def clear_mark(self):
        ind = int(self.last_pos.get())
        del self.marks[ind]
        if ind is self.mark_num: self.last_pos.set(int(self.last_pos.get()) - 1)
        self.mark_num = self.mark_num - 1
        self.mark_text.set('Out of {} marks'.format(self.mark_num))
        self.write_marks()

    def get_position(self):
        [x,y,z] = pyasi.get_positions(self.ser)
        self.x.set(x)
        self.y.set(y)
        self.z.set(z)
        self.root.after(20, self.get_position)

    def shut_down(self):
        self.ser.close()
        self.root.destroy()

    def new_mark_position(self):
        self.marks.append([self.x.get(),self.y.get(),self.z.get()])
        self.mark_num = self.mark_num + 1
        self.last_pos.set(self.mark_num)
        self.mark_text.set('Out of {} marks'.format(self.mark_num))
        self.write_marks()

    def update_mark(self):
        ind = int(self.last_pos.get())
        self.marks[ind] = [self.x.get(),self.y.get(),self.z.get()]
        

    def Start_Serial(self, ser):
        # Serial Communication Initialization
        self.ser = serial.Serial(ser, 9600)

    def send_command(self):
        self.response.set(pyasi.send_command(self.ser,self.cmd.get() + '\r'))

    def write_marks(self):
        f = open('positions.csv', 'w+')
        for i in range(len(self.marks)):
            f.write(str(self.marks[i][0]) + ',' + str(self.marks[i][1]) + ',' + str(self.marks[i][2])+'\n')
        f.close()


root = tk.Tk()

my_stage = stage_gui(root,'COM2')
root.mainloop()
