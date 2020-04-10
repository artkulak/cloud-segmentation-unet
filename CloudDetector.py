from tkinter import *
from PIL import ImageTk, Image
import tkinter.filedialog as tkFileDialog
import tkinter.ttk as ttk
from ttkthemes import themed_tk as tk
import threading
import cv2
import Model
import numpy as np
import matplotlib.pyplot as plt
import zipfile as zf

n_bands = 7
size = (256, 256)
path = '../valdata/'



class MainWindow:
    def __init__(self):
        self.scroll_index = 0


    @staticmethod
    def IoU(bboxes1, bboxes2):
        bboxes1 = np.array(bboxes1, dtype=bool)
        bboxes2 = np.array(bboxes2, dtype=bool)
        overlap = bboxes1 * bboxes2  # Logical AND
        union = bboxes1 + bboxes2  # Logical OR
        IOU = overlap.sum() / float(union.sum())

        return IOU

    @staticmethod
    def accuracy_metric(bboxes1, bboxes2):
        return np.sum(np.sum(bboxes1 == bboxes2, axis = 1))/bboxes1.shape[0]**2

    def progress(self):
        self.progress_bar['value'] += 100 / 16

    def LoadFile(self, ev):
        fn = tkFileDialog.Open(self.root).show(initialdir = path)
        if fn == '':
            return

        self.bands = np.empty((size[0], size[1], n_bands))
        with zf.ZipFile(fn, 'r') as zp:
            zp.extractall(path)

        self.mask = cv2.imread(path + 'landsat8/valdata/' + 'QB.tif')[:, :, 0]
        for band in range(n_bands):
            self.bands[:, :, band] = cv2.imread(path + 'landsat8/valdata/'+ f'B{band + 1}.tif')[:, :, 0]

        self.unmarked_image = ImageTk.PhotoImage(Image.fromarray(self.mask).resize((500, 500)))
        self.band.config(text="Mask")
        self.leftImFrame.create_image(25, 5, anchor=NW, image=self.unmarked_image)

    def SaveFile(self, ev):
        dir = tkFileDialog.askdirectory()
        if dir == '':
            return
        img = Image.fromarray(self.marked_image[:, :, 0].reshape((256, 256)))
        img.save(dir + '/mask.tif')

    def process_start(self, ev):
        self.th = threading.Thread(target=self.Process, args=(ev,), daemon=True)
        self.th.start()

    def scroll_left(self, ev):
        self.scroll_index -= 1
        if self.scroll_index < 0:
            self.scroll_index = n_bands + 1
            self.band.config(text=f"Band {7}")
            self.unmarked_image = ImageTk.PhotoImage(Image.fromarray(self.bands[:, :, -1]).resize((500, 500)))
        elif self.scroll_index == 0:
            self.band.config(text="Mask")
            self.unmarked_image = ImageTk.PhotoImage(Image.fromarray(self.mask).resize((500, 500)))
        else:
            self.band.config(text=f"Band {self.scroll_index}")
            self.unmarked_image = ImageTk.PhotoImage(
                Image.fromarray(self.bands[:, :, self.scroll_index - 1]).resize((500, 500)))

        self.leftImFrame.create_image(25, 5, anchor=NW, image=self.unmarked_image)

    def scroll_right(self, ev):
        self.scroll_index += 1
        if self.scroll_index >= n_bands + 1 or self.scroll_index == 0:
            self.scroll_index = 0
            self.band.config(text="Mask")
            self.unmarked_image = ImageTk.PhotoImage(Image.fromarray(self.mask).resize((500, 500)))
        else:
            self.band.config(text=f"Band {self.scroll_index}")
            self.unmarked_image = ImageTk.PhotoImage(
                Image.fromarray(self.bands[:, :, self.scroll_index - 1]).resize((500, 500)))

        self.leftImFrame.create_image(25, 5, anchor=NW, image=self.unmarked_image)

    def Process(self, ev):
        self.progress_bar['value'] = 0
        self.leftImFrame.create_image(7, 7, anchor=NW, image=None)
        self.marked_image = Model.Model().processSingle(self, self.bands)

        tmp = (np.repeat(self.marked_image[:, :, 0] > 0.1, 3).reshape((256, 256, 3))).astype(np.float32)
        self.marked_image = tmp
        self.iou.config(text=f'IOU {round(self.IoU((self.mask/255), tmp[:, :, 0].reshape((256, 256))), 3)}')
        self.accuracy.config(text = f'Accuracy {round(self.accuracy_metric((self.mask/255), tmp[:, :, 0].reshape((256, 256))), 3)}')
        plt.imsave('temp.png', tmp)
        self.temp = Image.open('temp.png')
        self.temp2 = ImageTk.PhotoImage(self.temp.resize((500, 500)))
        self.rightImFrame.create_image(15, 5, anchor = NW, image=self.temp2)

    def Run(self):
        self.root = tk.ThemedTk()
        self.root.get_themes()
        self.root.set_theme("default")
        s = ttk.Style()
        s.configure("green.Horizontal.TProgressbar", background='#0EFA74')
        self.root.geometry('1100x730')
        self.root.resizable(False, False)
        self.root.iconbitmap(r'logo.ico')
        self.root.title('CloudDetector')

        self.panelFrame = Canvas(self.root, bg = '#C4DDF2', height = 75, width = 1000).grid(row = 0, columnspan = 2, sticky = NSEW)

        # create main buttons
        load_icon = PhotoImage(file='load.png')
        save_icon = PhotoImage(file='save.png')
        detect_icon = PhotoImage(file='detect.png')

        self.loadBtn = Button(self.panelFrame, image=load_icon, relief=GROOVE)
        self.saveBtn = Button(self.panelFrame, image=save_icon, relief=GROOVE)
        self.processBtn = Button(self.panelFrame, image=detect_icon, relief=GROOVE)

        self.loadBtn.bind("<Button-1>", self.LoadFile)
        self.saveBtn.bind("<Button-1>", self.SaveFile)
        self.processBtn.bind("<Button-1>", self.process_start)

        self.loadBtn.grid(row = 0, column = 0, sticky = W, padx = 10)
        self.saveBtn.grid(row = 0, column = 0, sticky = W, padx = 160)
        self.processBtn.grid(row = 0, column = 1, sticky = E, padx = 20)


        self.leftImFrame = Canvas(self.root, height=500, width=500, bg="#123A52")
        self.leftImFrame.grid(row=1, column=0, sticky = NSEW)
        self.rightImFrame = Canvas(self.root, height=500, width=500, bg="#123A52")
        self.rightImFrame.grid(row=1, column=1, padx = 5, sticky = NSEW)

        left_arrow = PhotoImage(file='left.png')
        right_arrow = PhotoImage(file='right.png')
        self.band = Label(self.root, text='', font=("Arial", 16))

        leftBtn = Button(self.root, image=left_arrow, relief=GROOVE)
        rightBtn = Button(self.root, image=right_arrow, relief=GROOVE)

        leftBtn.bind("<Button-1>", self.scroll_left)
        rightBtn.bind("<Button-1>", self.scroll_right)

        leftBtn.grid(row=2, column=0, sticky = W, padx = 150)
        rightBtn.grid(row = 2, column = 0, sticky = E, padx = 150)
        self.band.grid(row = 2, column = 0)

        self.accuracy = Label(self.root, text='Accuracy: 0 %', font=("Arial", 12))
        self.accuracy.grid(row = 2, column = 1, sticky = N, pady = 5)
        self.iou = Label(self.root, text='IoU: 0 %', font=("Arial", 12))
        self.iou.grid(row = 2, column = 1, sticky = S, pady = 10)

        self.progress_bar = ttk.Progressbar(self.root, style = 'green.Horizontal.TProgressbar', mode="determinate", orient='horizontal', length = 1290)
        self.progress_bar.grid(row = 3, columnspan = 2, ipady = 15, padx = 5, pady = 5,  sticky = NSEW)

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)

        self.root.mainloop()


if __name__ == '__main__':
    window = MainWindow()
    window.Run()
