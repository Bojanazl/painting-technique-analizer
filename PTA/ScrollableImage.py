# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 23:56:41 2020

@author: Bojana
"""

import tkinter
from PIL import ImageTk, Image

class ScrollableImage(tkinter.Canvas):
    def __init__(self, master=None, **kw):
        self.image = kw.pop('image', None)
        super(ScrollableImage, self).__init__(master=master, **kw)
        self['highlightthickness'] = 0
        self.propagate(0)  # wont let the scrollbars rule the size of Canvas
        self.create_image(0,0, anchor='nw', image=self.image)
        # Vertical and Horizontal scrollbars
        self.v_scroll = tkinter.Scrollbar(self, orient='vertical', width=6)
        self.h_scroll = tkinter.Scrollbar(self, orient='horizontal', width=6)
        self.v_scroll.pack(side='right', fill='y')
        self.h_scroll.pack(side='bottom', fill='x')
        # Set the scrollbars to the canvas
        self.config(xscrollcommand=self.h_scroll.set, 
                yscrollcommand=self.v_scroll.set)
        # Set canvas view to the scrollbars
        self.v_scroll.config(command=self.yview)
        self.h_scroll.config(command=self.xview)
        # Assign the region to be scrolled 
        self.config(scrollregion=self.bbox('all'))

        self.focus_set()
        self.bind_class(self, "<MouseWheel>", self.mouse_scroll)

    def mouse_scroll(self, evt):
        if evt.state == 0 :
            # self.yview_scroll(-1*(evt.delta), 'units') # For MacOS
            self.yview_scroll( int(-1*(evt.delta/120)) , 'units') # For windows
        if evt.state == 1:
            # self.xview_scroll(-1*(evt.delta), 'units') # For MacOS
            self.xview_scroll( int(-1*(evt.delta/120)) , 'units') # For windows

# --------- Example and testing ---------
# This will be true only if the file is not imported as a package.
if __name__ == "__main__":      
    root = tkinter.Tk()
    img = Image.open('test.jpg')
    img = ImageTk.PhotoImage(img)
    ScrollableImage(root, image=img, width=200, height=200).pack()
    root.mainloop()