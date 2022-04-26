# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:40:05 2020

@author: Bojana
"""
#C:\Fakultet\Multimedijalni sistemi\Projektni\Radovi
#Canvas omogucava crtanje 



from tkinter import *
from PIL import ImageTk,Image 
from tkinter import filedialog
from scrollimage import ScrollableImage

root=Tk()
root.geometry('600x800')
root.title('World is the Canvas')
path="C:\Fakultet\Multimedijalni sistemi\Projektni\Radovi"
root.iconbitmap('C:\Fakultet\Multimedijalni sistemi\Projektni\icon.png') #ne radi

#functions
def loadImage():
    global my_image
    root.filename=filedialog.askopenfilename(initialdir="path", title="Load an Image", filetypes=(("all files", "*.*"),("png files", "*.png"),("jpg files", "*.jpg")))
    my_label=Label(root, text=root.filename).pack()
    my_image=ImageTk.PhotoImage(Image.open(root.filename))
    my_image_label=Label(image=my_image).pack()
    
#frame
#frame=Frame(root)
#frame.grid(side=RIGHT)
#frametwo=Frame(root)
#frametwo.pack(side=LEFT)


#buttons
my_btn=Button(root, text="Load an Image", command=loadImage).pack()

#sliders vert and horizont
#vertical=Scale(root, from_=0, to=200)
#vertical.grid(row=0, column=1)

#horizontal=Scale(root, from_=0, to=200, orient=horizontal)
#horizontal.gird()



#kreiramo event
root.mainloop()