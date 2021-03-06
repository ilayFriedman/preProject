#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.19
#  in conjunction with Tcl version 8.6
#    Apr 23, 2019 10:38:47 AM +0300  platform: Windows NT
from tkinter import messagebox

from flower_classification import *
import sys
from tkinter.filedialog import askdirectory, askopenfilename, IntVar

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()

    top = Toplevel1 (root)
    root.mainloop()

w = None
def create_Toplevel1(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)

    top = Toplevel1 (w)

    return (w, top)

def destroy_Toplevel1():
    global w
    w.destroy()
    w = None

class Toplevel1:
    def __init__(self, top=None):

        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'
        font9 = "-family {Segoe UI} -size 12 -weight normal -slant "  \
            "roman -underline 0 -overstrike 0"
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.',background=_bgcolor)
        self.style.configure('.',foreground=_fgcolor)
        self.style.configure('.',font="TkDefaultFont")
        self.style.map('.',background=
            [('selected', _compcolor), ('active',_ana2color)])

        top.geometry("600x450+437+267")
        top.title("New Toplevel")
        top.configure(background="#d9d9d9")

        self.input_Frame = ttk.Frame(top)
        self.input_Frame.place(relx=0.017, rely=0.022, relheight=0.189
                , relwidth=0.742)
        self.input_Frame.configure(relief='groove')
        self.input_Frame.configure(borderwidth="2")
        self.input_Frame.configure(relief='groove')
        self.input_Frame.configure(width=445)


        self.ImgBrowseButton = ttk.Button(self.input_Frame)
        self.ImgBrowseButton.place(relx=0.798, rely=0.059, height=25, width=76)
        self.ImgBrowseButton.configure(takefocus="")
        self.ImgBrowseButton.configure(text='''Browse''')
        self.ImgBrowseButton.configure(command=self.ImagePathBrowse)

        self.ModelBrowseButton = ttk.Button(self.input_Frame)
        self.ModelBrowseButton.place(relx=0.798, rely=0.471, height=25, width=76)

        self.ModelBrowseButton.configure(takefocus="")
        self.ModelBrowseButton.configure(text='''Browse''')
        self.ModelBrowseButton.configure(command=self.ModelBrowse)

        self.ImagePathTextField = ttk.Entry(self.input_Frame)
        self.ImagePathTextField.place(relx=0.18, rely=0.118, relheight=0.247
                , relwidth=0.598)
        self.ImagePathTextField.configure(width=266)
        self.ImagePathTextField.configure(takefocus="")
        self.ImagePathTextField.configure(cursor="ibeam")

        self.ImagePathLabel = ttk.Label(self.input_Frame)
        self.ImagePathLabel.place(relx=0.022, rely=0.118, height=19, width=67)
        self.ImagePathLabel.configure(background="#d9d9d9")
        self.ImagePathLabel.configure(foreground="#000000")
        self.ImagePathLabel.configure(font="TkDefaultFont")
        self.ImagePathLabel.configure(relief='flat')
        self.ImagePathLabel.configure(text='''Image Path:''')

        self.ModelPathTextField = ttk.Entry(self.input_Frame)
        self.ModelPathTextField.place(relx=0.18, rely=0.471, relheight=0.247
                , relwidth=0.598)
        self.ModelPathTextField.configure(width=266)
        self.ModelPathTextField.configure(takefocus="")
        self.ModelPathTextField.configure(cursor="ibeam")

        self.ModelPathLabel = ttk.Label(self.input_Frame)
        self.ModelPathLabel.place(relx=0.022, rely=0.471, height=19, width=68)
        self.ModelPathLabel.configure(background="#d9d9d9")
        self.ModelPathLabel.configure(foreground="#000000")
        self.ModelPathLabel.configure(font="TkDefaultFont")
        self.ModelPathLabel.configure(relief='flat')
        self.ModelPathLabel.configure(text='''Model Path:''')

        self.tileClass = tk.Label(top)
        self.tileClass.place(relx=0.792, rely=0.0, height=27, width=97)
        self.tileClass.configure(background="#d9d9d9")
        self.tileClass.configure(disabledforeground="#a3a3a3")
        self.tileClass.configure(font=font9)
        self.tileClass.configure(foreground="#000000")
        self.tileClass.configure(text='''Classification''')

        self.PredictButton = ttk.Button(top)
        self.PredictButton.place(relx=0.792, rely=0.067, height=25, width=106)
        self.PredictButton.configure(takefocus="")
        self.PredictButton.configure(text='''Predict!''')
        self.PredictButton.configure(width=106)
        self.PredictButton.configure(state='disabled')
        self.PredictButton.configure(command=self.predictAction)

        self.ResetButton = ttk.Button(top)
        self.ResetButton.place(relx=0.792, rely=0.133, height=25, width=106)
        self.ResetButton.configure(takefocus="")
        self.ResetButton.configure(text='''Reset All''')
        self.ResetButton.configure(width=106)
        self.ResetButton.configure(command=self.resetAll)

        self.DaisyCheckB = tk.Checkbutton(top)
        self.DaisyCheckB.place(relx=0.017, rely=0.333, relheight=0.056
                , relwidth=0.093)
        self.DaisyCheckB.configure(activebackground="#ececec")
        self.DaisyCheckB.configure(activeforeground="#000000")
        self.DaisyCheckB.configure(background="#d9d9d9")
        self.DaisyCheckB.configure(disabledforeground="#a3a3a3")
        self.DaisyCheckB.configure(foreground="#000000")
        self.DaisyCheckB.configure(highlightbackground="#d9d9d9")
        self.DaisyCheckB.configure(highlightcolor="black")
        self.DaisyCheckB.configure(justify='left')
        self.DaisyCheckB.configure(text='''Daisy''')
        self.DaisyCheck = IntVar()
        self.DaisyCheckB.configure(variable=self.DaisyCheck)
        #self.DaisyCheckB.configure(variable=unknown_support.che57)



        self.DandelionCheckB = tk.Checkbutton(top)
        self.DandelionCheckB.place(relx=0.017, rely=0.422, relheight=0.056
                , relwidth=0.137)
        self.DandelionCheckB.configure(activebackground="#ececec")
        self.DandelionCheckB.configure(activeforeground="#000000")
        self.DandelionCheckB.configure(background="#d9d9d9")
        self.DandelionCheckB.configure(disabledforeground="#a3a3a3")
        self.DandelionCheckB.configure(foreground="#000000")
        self.DandelionCheckB.configure(highlightbackground="#d9d9d9")
        self.DandelionCheckB.configure(highlightcolor="black")
        self.DandelionCheckB.configure(justify='left')
        self.DandelionCheckB.configure(text='''Dandelion''')
        self.DandelionCheck = IntVar()
        self.DandelionCheckB.configure(variable=self.DandelionCheck)
        #self.DandelionCheckB.configure(variable=unknown_support.che58)

        self.RoseCheckB = tk.Checkbutton(top)
        self.RoseCheckB.place(relx=0.017, rely=0.378, relheight=0.056
                , relwidth=0.088)
        self.RoseCheckB.configure(activebackground="#ececec")
        self.RoseCheckB.configure(activeforeground="#000000")
        self.RoseCheckB.configure(background="#d9d9d9")
        self.RoseCheckB.configure(disabledforeground="#a3a3a3")
        self.RoseCheckB.configure(foreground="#000000")
        self.RoseCheckB.configure(highlightbackground="#d9d9d9")
        self.RoseCheckB.configure(highlightcolor="black")
        self.RoseCheckB.configure(justify='left')
        self.RoseCheckB.configure(text='''Rose''')
        self.RoseCheck = IntVar()
        self.RoseCheckB.configure(variable=self.RoseCheck)
        #self.RoseCheckB.configure(variable=unknown_support.che59)

        self.SunflowerCheckB = tk.Checkbutton(top)
        self.SunflowerCheckB.place(relx=0.017, rely=0.467, relheight=0.056
                , relwidth=0.135)
        self.SunflowerCheckB.configure(activebackground="#ececec")
        self.SunflowerCheckB.configure(activeforeground="#000000")
        self.SunflowerCheckB.configure(background="#d9d9d9")
        self.SunflowerCheckB.configure(disabledforeground="#a3a3a3")
        self.SunflowerCheckB.configure(foreground="#000000")
        self.SunflowerCheckB.configure(highlightbackground="#d9d9d9")
        self.SunflowerCheckB.configure(highlightcolor="black")
        self.SunflowerCheckB.configure(justify='left')
        self.SunflowerCheckB.configure(text='''Sunflower''')
        self.SunflowerCheck = IntVar()
        self.SunflowerCheckB.configure(variable=self.SunflowerCheck)
        #self.SunflowerCheckB.configure(variable=unknown_support.che60)

        self.tulipCheckB = tk.Checkbutton(top)
        self.tulipCheckB.place(relx=0.017, rely=0.511, relheight=0.056, relwidth=0.092)
        self.tulipCheckB.configure(activebackground="#ececec")
        self.tulipCheckB.configure(activeforeground="#000000")
        self.tulipCheckB.configure(background="#d9d9d9")
        self.tulipCheckB.configure(disabledforeground="#a3a3a3")
        self.tulipCheckB.configure(foreground="#000000")
        self.tulipCheckB.configure(highlightbackground="#d9d9d9")
        self.tulipCheckB.configure(highlightcolor="black")
        self.tulipCheckB.configure(justify='left')
        self.tulipCheckB.configure(text='''Tulip''')
        self.tulipCheck = IntVar()
        self.tulipCheckB.configure(variable=self.tulipCheck)

        self.filterButton = ttk.Button(top)
        self.filterButton.place(relx=0.017, rely=0.565, height=25, width=76)
        self.filterButton.configure(takefocus="")
        self.filterButton.configure(text='''Filter By''')
        self.filterButton.configure(command=self.filterBy)


        self.ShowAll = ttk.Button(top)
        self.ShowAll.place(relx=0.017, rely=0.622, height=25, width=106)
        self.ShowAll.configure(takefocus="")
        self.ShowAll.configure(text='''Show All Types''')
        self.ShowAll.configure(command=self.showAll)



        self.TLabel3 = ttk.Label(top)
        self.TLabel3.place(relx=0.017, rely=0.222, height=25, width=54)
        self.TLabel3.configure(background="#d9d9d9")
        self.TLabel3.configure(foreground="#000000")
        self.TLabel3.configure(font=font9)
        self.TLabel3.configure(relief='flat')
        self.TLabel3.configure(text='''Results''')

        self.TLabel4 = ttk.Label(top)
        self.TLabel4.place(relx=0.017, rely=0.289, height=19, width=112)
        self.TLabel4.configure(background="#d9d9d9")
        self.TLabel4.configure(foreground="#000000")
        self.TLabel4.configure(font="TkDefaultFont")
        self.TLabel4.configure(relief='flat')
        self.TLabel4.configure(text='''Show in results only:''')

        self.style.configure('Treeview.Heading', font="TkDefaultFont")
        self.Scrolledtreeview1 = ScrolledTreeView(top)
        self.Scrolledtreeview1.place(relx=0.367, rely=0.244, relheight=0.66
                                     , relwidth=0.59)
        self.Scrolledtreeview1.configure(columns="Col1 Col2")
        self.Scrolledtreeview1.heading("#0", text="")
        self.Scrolledtreeview1.heading("#0", anchor="center")
        self.Scrolledtreeview1.column("#0", width="0")
        self.Scrolledtreeview1.column("#0", stretch="1")
        self.Scrolledtreeview1.column("#0", anchor="w")
        self.Scrolledtreeview1.heading("Col1", text="File Name")
        self.Scrolledtreeview1.heading("Col1", anchor="center")
        self.Scrolledtreeview1.column("Col1", width="167")
        self.Scrolledtreeview1.column("Col1", minwidth="20")
        self.Scrolledtreeview1.column("Col1", stretch="1")
        self.Scrolledtreeview1.column("Col1", anchor="w")

        self.Scrolledtreeview1.heading("Col2", text="Classification")
        self.Scrolledtreeview1.heading("Col2", anchor="center")
        self.Scrolledtreeview1.column("Col2", width="167")
        self.Scrolledtreeview1.column("Col2", minwidth="20")
        self.Scrolledtreeview1.column("Col2", stretch="1")
        self.Scrolledtreeview1.column("Col2", anchor="w")
        self.DaisyCheckB.configure(state='disable')
        self.DandelionCheckB.configure(state='disable')
        self.SunflowerCheckB.configure(state='disable')
        self.RoseCheckB.configure(state='disable')
        self.tulipCheckB.configure(state='disable')
        self.ShowAll.configure(state='disable')
        self.filterButton.configure(state='disable')

    def ImagePathBrowse(self):
        dirWind = tk.Tk()
        dirWind.withdraw()
        path = askdirectory()
        if (len(str(self.ImagePathTextField.get())) != 0):
            self.ImagePathTextField.delete(0, 'end')
        self.ImagePathTextField.insert(0, str(path))
        dirWind.destroy()
        if((len(str(self.ImagePathTextField.get()))!= 0) and (len(str(self.ModelPathTextField.get())) != 0)):
            self.PredictButton.configure(state='normal')
    def ModelBrowse(self):
        dirWind = tk.Tk()
        dirWind.withdraw()
        path = askopenfilename();
        if(len(str(self.ModelPathTextField.get())) != 0):
            self.ModelPathTextField.delete(0,'end')
        self.ModelPathTextField.insert(0,str(path))
        dirWind.destroy()
        if((len(str(self.ImagePathTextField.get()))!= 0) and (len(str(self.ModelPathTextField.get())) != 0)):
            self.PredictButton.configure(state='normal')
    def resetAll(self):
        self.ModelPathTextField.delete(0, 'end')
        self.ImagePathTextField.delete(0, 'end')
        self.PredictButton.configure(state='disable')
        self.Scrolledtreeview1.delete(*self.Scrolledtreeview1.get_children())
        self.DaisyCheckB.configure(state='disable')
        self.DandelionCheckB.configure(state='disable')
        self.SunflowerCheckB.configure(state='disable')
        self.RoseCheckB.configure(state='disable')
        self.tulipCheckB.configure(state='disable')
        self.ImagePathTextField.configure(state='normal')
        self.ModelPathTextField.configure(state='normal')
        self.ModelBrowseButton.configure(state='normal')
        self.ImgBrowseButton.configure(state='normal')
        self.PredictButton.configure(state='normal')
        self.ShowAll.configure(state='disable')
        self.filterButton.configure(state='disable')
        self.DaisyCheck.set(0)
        self.RoseCheck.set(0)
        self.SunflowerCheck.set(0)
        self.tulipCheck.set(0)
        self.DandelionCheck.set(0)
        self.ModelPathTextField.delete(0, 'end')
        self.ImagePathTextField.delete(0, 'end')


    def predictAction(self):
        if(self.ModelPathTextField.get().endswith(".h5") != True):
            messagebox.showerror('oops!', 'Please insert a valid Model-path! (.h5 File)')
        elif ( len(os.listdir(self.ImagePathTextField.get()) ) == 0 ):
                messagebox.showerror('oops!', 'That folder is empty! Theres nothing to predict!')
        else:
            self.ImagePathTextField.configure(state='disable')
            self.ModelPathTextField.configure(state='disable')
            self.ModelBrowseButton.configure(state='disable')
            self.ImgBrowseButton.configure(state='disable')
            self.PredictButton.configure(state='disable')
            try:
                self.names,self.answer = predict(load_trained_model(self.ModelPathTextField.get()), self.ImagePathTextField.get())
                for i in range(len(self.names)):
                    #print(names[i],answer[i])
                    self.Scrolledtreeview1.insert("",'end',i+1,values=(self.names[i],self.answer[i]))
                self.DaisyCheckB.configure(state='normal')
                self.DandelionCheckB.configure(state='normal')
                self.SunflowerCheckB.configure(state='normal')
                self.RoseCheckB.configure(state='normal')
                self.tulipCheckB.configure(state='normal')
                self.ShowAll.configure(state='normal')
                self.filterButton.configure(state='normal')
            except:
                messagebox.showerror('oops!', 'Something did not work out. check your files please and try again!')
    def filterBy(self):
        self.filterType=[]
        if(self.DaisyCheck.get() == 1):
            self.filterType.append("daisy")
        if(self.RoseCheck.get() == 1):
            self.filterType.append("rose")
        if(self.SunflowerCheck.get() == 1):
            self.filterType.append("sunflower")
        if(self.tulipCheck.get() == 1):
            self.filterType.append("tulip")
        if(self.DandelionCheck.get() == 1):
            self.filterType.append("dandelion")
        self.Scrolledtreeview1.delete(*self.Scrolledtreeview1.get_children())
        for i in range(len(self.names)):
            if(self.answer[i] in self.filterType):
                self.Scrolledtreeview1.insert("", 'end', i + 1, values=(self.names[i], self.answer[i]))
        self.filterType.clear()

    def showAll(self):
        self.Scrolledtreeview1.delete(*self.Scrolledtreeview1.get_children())
        for i in range(len(self.names)):
            # print(names[i],answer[i])
            self.Scrolledtreeview1.insert("", 'end', i + 1, values=(self.names[i], self.answer[i]))
        self.DaisyCheck.set(0)
        self.RoseCheck.set(0)
        self.SunflowerCheck.set(0)
        self.tulipCheck.set(0)
        self.DandelionCheck.set(0)



# The following code is added to facilitate the Scrolled widgets you specified.
class AutoScroll(object):
    '''Configure the scrollbars for a widget.'''

    def __init__(self, master):
        #  Rozen. Added the try-except clauses so that this class
        #  could be used for scrolled entry widget for which vertical
        #  scrolling is not supported. 5/7/14.
        try:
            vsb = ttk.Scrollbar(master, orient='vertical', command=self.yview)
        except:
            pass
        hsb = ttk.Scrollbar(master, orient='horizontal', command=self.xview)

        #self.configure(yscrollcommand=_autoscroll(vsb),
        #    xscrollcommand=_autoscroll(hsb))
        try:
            self.configure(yscrollcommand=self._autoscroll(vsb))
        except:
            pass
        self.configure(xscrollcommand=self._autoscroll(hsb))

        self.grid(column=0, row=0, sticky='nsew')
        try:
            vsb.grid(column=1, row=0, sticky='ns')
        except:
            pass
        hsb.grid(column=0, row=1, sticky='ew')

        master.grid_columnconfigure(0, weight=1)
        master.grid_rowconfigure(0, weight=1)

        # Copy geometry methods of master  (taken from ScrolledText.py)
        if py3:
            methods = tk.Pack.__dict__.keys() | tk.Grid.__dict__.keys() \
                  | tk.Place.__dict__.keys()
        else:
            methods = tk.Pack.__dict__.keys() + tk.Grid.__dict__.keys() \
                  + tk.Place.__dict__.keys()

        for meth in methods:
            if meth[0] != '_' and meth not in ('config', 'configure'):
                setattr(self, meth, getattr(master, meth))

    @staticmethod
    def _autoscroll(sbar):
        '''Hide and show scrollbar as needed.'''
        def wrapped(first, last):
            first, last = float(first), float(last)
            if first <= 0 and last >= 1:
                sbar.grid_remove()
            else:
                sbar.grid()
            sbar.set(first, last)
        return wrapped

    def __str__(self):
        return str(self.master)

def _create_container(func):
    '''Creates a ttk Frame with a given master, and use this new frame to
    place the scrollbars and the widget.'''
    def wrapped(cls, master, **kw):
        container = ttk.Frame(master)
        container.bind('<Enter>', lambda e: _bound_to_mousewheel(e, container))
        container.bind('<Leave>', lambda e: _unbound_to_mousewheel(e, container))
        return func(cls, container, **kw)
    return wrapped

class ScrolledTreeView(AutoScroll, ttk.Treeview):
    '''A standard ttk Treeview widget with scrollbars that will
    automatically show/hide as needed.'''
    @_create_container
    def __init__(self, master, **kw):
        ttk.Treeview.__init__(self, master, **kw)
        AutoScroll.__init__(self, master)

import platform
def _bound_to_mousewheel(event, widget):
    child = widget.winfo_children()[0]
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        child.bind_all('<MouseWheel>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-MouseWheel>', lambda e: _on_shiftmouse(e, child))
    else:
        child.bind_all('<Button-4>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Button-5>', lambda e: _on_mousewheel(e, child))
        child.bind_all('<Shift-Button-4>', lambda e: _on_shiftmouse(e, child))
        child.bind_all('<Shift-Button-5>', lambda e: _on_shiftmouse(e, child))

def _unbound_to_mousewheel(event, widget):
    if platform.system() == 'Windows' or platform.system() == 'Darwin':
        widget.unbind_all('<MouseWheel>')
        widget.unbind_all('<Shift-MouseWheel>')
    else:
        widget.unbind_all('<Button-4>')
        widget.unbind_all('<Button-5>')
        widget.unbind_all('<Shift-Button-4>')
        widget.unbind_all('<Shift-Button-5>')

def _on_mousewheel(event, widget):
    if platform.system() == 'Windows':
        widget.yview_scroll(-1*int(event.delta/120),'units')
    elif platform.system() == 'Darwin':
        widget.yview_scroll(-1*int(event.delta),'units')
    else:
        if event.num == 4:
            widget.yview_scroll(-1, 'units')
        elif event.num == 5:
            widget.yview_scroll(1, 'units')

def _on_shiftmouse(event, widget):
    if platform.system() == 'Windows':
        widget.xview_scroll(-1*int(event.delta/120), 'units')
    elif platform.system() == 'Darwin':
        widget.xview_scroll(-1*int(event.delta), 'units')
    else:
        if event.num == 4:
            widget.xview_scroll(-1, 'units')
        elif event.num == 5:
            widget.xview_scroll(1, 'units')

if __name__ == '__main__':
    vp_start_gui()





