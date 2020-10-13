#Sinais e Multimídia - Projeto Final
#Aplicativo de Processamento de Imagens utilizando convoluções de kernels sobre imagens do webcam com detecção de blur

#Integrantes do grupo - Matrículas

#Vinícius Pereira Duarte - 11721ECP003
#Vitor Rabelo Cruvinel - 11721ECP004
#Renato Junio Martins - 11721ECP003

import time
import random
import numpy as np
import cv2
import tkinter as tk
import tkinter.font as tkFont
from scipy import signal, ndimage
import timeit
from PIL import Image, ImageTk
import importlib

#Testing imports
if importlib.util.find_spec("cv2") is None:
    print("opencv-python isn't installed. Try 'pip3 install opencv-python'")
if importlib.util.find_spec("PIL") is None:
    print("Pillow isn't installed. Try 'pip3 install Pillow'")
if importlib.util.find_spec("tkinter") is None:
    print("Tkinter isn't installed. Try 'sudo apt install python3-tk'")

class MyApp:
    def __init__(self, root):
        self.root = root

        # Getting screen resolution
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.root.minsize(width=int(self.screen_width*.8), height=int(self.screen_height*.8))

        # Configuring fonts
        font_size = 13 if self.screen_width <= 1366 else 15
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=font_size)

        # Vars
        self.checked = tk.BooleanVar()
        self.option = tk.StringVar()
        self.blur_thresh = tk.DoubleVar()
        self.blur_thresh.set(9)
        self.test_convolution = None
        self.cap = cv2.VideoCapture(0)
        self.last_frametime = time.time()
        self.frametime_sum_tk = 0
        self.n_frame = 0
        self.mean_size = 100

        # Available kernels
        self.kernel = {
            'identity': np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float),
            'edge detection': np.array([
                [1,0,-1],
                [0,0,0],
                [-1,0,1]
            ], dtype=float),
            'laplacian': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float),
            'laplacian w/ diagonals': np.array([
                [-1,-1,-1],
                [-1,8,-1],
                [-1,-1,-1]
            ], dtype=float),
            'laplacian of gaussian': np.array([
                [0,0,-1,0,0],
                [0,-1,-2,-1,0],
                [-1,-2,16,-2,-1],
                [0,-1,-2,-1,0],
                [0,0,-1,0,0]
            ], dtype=float),
            'scharr': np.array([
                [-3, 0, 3],
                [-10,0,10],
                [-3, 0, 3]
            ], dtype=float),
            'sobel edge horizontal': np.array([
                [-1,-2,-1],
                [0,0,0],
                [1,2,1]
            ], dtype=float),
            'sobel edge vertical': np.array([
                [-1,0,1],
                [-2,0,2],
                [-1,0,1]
            ], dtype=float),
            'line detection horizontal': np.array([
                [-1,-1,-1],
                [2,2,2],
                [-1,-1,-1]
            ], dtype=float),
            'line detection vertical': np.array([
                [-1,2,-1],
                [-1,2,-1],
                [-1,2,-1]
            ], dtype=float),
            'line detection 45°': np.array([
                [-1,-1,2],
                [-1,2,-1],
                [2,-1,-1]
            ], dtype=float),
            'line detection 135°': np.array([
                [2,-1,-1],
                [-1,2,-1],
                [-1,-1,2]
            ], dtype=float),
            'box blur': (1/9)*np.ones((3,3), dtype=float),
            'gaussian blur 3x3': (1/16)*np.array([
                [1,2,1],
                [2,4,2],
                [1,2,1]
            ], dtype=float),
            'gaussian blur 5x5': (1/256)*np.array([
                [1,4,6,4,1],
                [4,16,24,16,4],
                [6,24,36,24,6],
                [4,16,24,16,4],
                [1,4,6,4,1]
            ], dtype=float),
            'sharpen': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
            'unsharp masking': (-1/256)*np.array([
                [1,4,6,4,1],
                [4,16,24,16,4],
                [6,24,-476,24,6],
                [4,16,24,16,4],
                [1,4,6,4,1]
            ], dtype=float),
        }

        #Getting kernel keys and setting first option
        self.kernel_keys = self.kernel.keys()
        self.option.set(list(self.kernel_keys)[0])

        #Frames of the application
        self.frame1 = tk.Frame(root, background="#282a36")
        self.frame1.pack(fill="both", expand=True, side="top")
        self.frame2 = tk.Frame(self.frame1, background="#282a36")
        self.frame2.pack(fill="x", expand=False, side="left", anchor="n")
        self.frame3 = tk.Frame(self.frame2, background="#282a36")
        self.frame3.pack(fill="x", expand=True, side="top", anchor="n", padx=5)
        self.frame4 = tk.Frame(self.frame1, background="#282a36")
        self.frame4.pack(fill="both", expand=True, side="right", anchor="n")

        #--------------Widgets--------------#
        #Status of blur detection
        self.status = tk.Label(
            self.frame2,
            text="Not blurred",
            background="#282a36",
            foreground="#f8f8f2",
        )

        # Values of blur
        self.blur_value = tk.Label(
            self.frame2,
            text="",
            background="#282a36",
            foreground="#f8f8f2",
        )

        # Threshold adjust label
        self.label_threshold = tk.Label(
            self.frame2,
            text="Threshold adjust:",
            background="#282a36",
            foreground="#f8f8f2",
        )

        #Checkbutton of blur detection
        self.check_blur = tk.Checkbutton(
            self.frame3,
            text="Blur detection",
            variable=self.checked,
            pady=10,
            padx=5,
            command=self.showBlurOptions,
            background="#282a36",
            activebackground="#44475a",
            activeforeground="#f8f8f2",
            highlightcolor="#f8f8f2",
            borderwidth=1,
            highlightthickness=1,
            relief="solid",
            foreground="#f8f8f2",
            selectcolor="#44475a"
        )
        self.check_blur.pack(side="left")

        # Threshold scale
        self.scale = tk.Scale(
            self.frame2,
            variable=self.blur_thresh,
            orient="horizontal",
            background="#282a36",
            activebackground="#44475a",
            foreground="#f8f8f2",
            borderwidth=1,
            from_=-5,
            to=25
        )

        #Listbox of kernels
        self.listbox = tk.Listbox(
            self.frame2,
            background="#282a36",
            foreground="#f8f8f2",
            selectforeground="#f8f8f2",
            selectbackground="#44475a",
            borderwidth=1,
            highlightcolor="#f8f8f2",
            relief="solid",
            height=17
        )
        self.listbox.pack(side="bottom", pady=5, padx=5, fill="both", expand=True)

        # Binding selected kernel of listbox with option variable
        self.listbox.bind('<<ListboxSelect>>', self.onSelection)

        # Filling up the listbox
        for item in self.kernel_keys:
            self.listbox.insert("end", item)

        # Selecting first kernel
        self.listbox.select_set(0)

        #Button for convolution test
        self.button_test_convolution = tk.Button(
            self.frame3,
            text= "Test convolutions",
            background="#282a36",
            foreground="#f8f8f2",
            activebackground="#44475a",
            activeforeground="#f8f8f2",
            borderwidth=1,
            pady=9,
            highlightcolor="#f8f8f2",
            relief="solid",
            command = self.testLatestConvolution
        )
        self.button_test_convolution.pack(side="right", padx=(5,0))

        # Label of captured frames
        self.label_frame = tk.Label(self.frame4, borderwidth=0, background="#282a36")
        self.label_frame.pack()

    def onSelection(self, event):
        self.option.set(self.listbox.get(self.listbox.curselection()))

    # Dynamic Image resizing
    def imgSizeAdjust(self, img):
        stable_width = self.frame1.winfo_width() - self.frame2.winfo_width()
        ratio = stable_width/img.width
        stable_height = int(ratio*img.height)
        if (stable_width > 1):
            return img.resize((stable_width, stable_height), Image.NEAREST)
        else:
            return img.resize((img.width, img.height), Image.NEAREST)

    def showBlurOptions(self):
        if(self.checked.get() == True):
            self.status.pack(side="top", pady=(5,0))
            self.blur_value.pack(side="top")
            self.label_threshold.pack(side="top")
            self.scale.pack(side="top", pady=(0, 5), fill="x", padx=5)
        else:
            self.status.pack_forget()
            self.scale.pack_forget()
            self.blur_value.pack_forget()
            self.label_threshold.pack_forget()

    def detectBlur(self, im, cut_freq=60, thresh=None):
        thresh = thresh or self.blur_thresh.get()
        fft = np.fft.fft2(im)
        (h, w) = fft.shape
        fft[0:cut_freq,0:cut_freq]     = 0
        fft[h-cut_freq:h,0:cut_freq]   = 0
        fft[0:cut_freq,w-cut_freq:w]   = 0
        fft[h-cut_freq:h,w-cut_freq:w] = 0
        recon = np.fft.ifft2(fft)
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        return (mean, thresh)

    def detectBlur_convolve(self, im):
        thresh = self.blur_thresh.get()
        mean = self.convolve(im, self.kernel['laplacian'], 5).var()/10 - 10
        return (mean, thresh)

    def setBlurred(self, tuple): #(value, maxValue)
        is_blurred = tuple[0] <= tuple[1]
        value = '%.2f'%tuple[0]
        max_value = '%.2f'%tuple[1]
        if(is_blurred):
            self.status.config(text="Blurred")
            self.blur_value.config(text="Value: {} of {}".format(value, max_value))
        else:
            self.status.config(text="Not blurred")
            self.blur_value.config(text="Value: {} of {}".format(value, max_value))

    # Array of convolutions implemented
    def getConvolutions(self):
        convolutions = [
            #convolve_pure,
            self.convolve_fft,
            self.convolve_scipy,
            self.convolve_scipy_fft,
            self.convolve_scipy_2d,
            self.convolve_uint_scipy,
            self.convolve_uint_scipy_view16,
            self.convolve_uint_scipy_view32
        ]
        return convolutions

    # The convolve method uses the fastest convolution implemented
    def convolve(self, im, omega, index):
        default_index = 1
        convolutions = self.getConvolutions()
        index = index or default_index
        if(index > len(convolutions)-1):
            print("Convolução com o index {} não encontrado, carregando default".format(index))
            index = default_index
        result = convolutions[index](im,omega)
        return result

    def testLatestConvolution(self):
        print("Testing convolutions with the last image")
        gray, omega = self.test_convolution
        self.testConvolutions(self.getConvolutions(), gray, omega, 1)

    # The testConvolutions calculates the time used for each convolution method
    def testConvolutions(self, convolutions, im, omega, inverse_chance=1):  #1 of inverse_chance of executing the test
        if(random.randint(1,inverse_chance) == 1):
            for i,convolution in enumerate(convolutions):
                number_of_executions = 100
                duration = timeit.timeit(lambda: convolution(im,omega), number=number_of_executions)
                print("Tempo de execução da convolução {index}: {duration:.3f}ms".format(index = i ,duration = duration*1000/number_of_executions))

    # Convolutions implementations

    def convolve_pure(self, im, omega):
        M, N = im.shape
        A, B = omega.shape
        a, b = A//2, B//2 # kernel with odd dimensions
        f = np.array(im, dtype=float)
        g = np.zeros_like(f, dtype=float)
        for x in range(M):
            for y in range(N):
                aux = 0.0
                for dx in range(-a, a+1):
                    for dy in range(-b, b+1):
                        # or you could use "zero pad" on image
                        if 0 <= x+dx < M and 0 <= y+dy < N:
                            aux += omega[a-dx, b-dy]*f[x+dx, y+dy]
                g[x, y] = aux
        return g

    def convolve_fft(self, im, omega):
        im = np.pad(im, ((0,1), (0,1))) # zero pad in the last row and column
        spi = np.fft.fft2(im)
        spf = np.fft.fft2(omega, s=im.shape)
        g = spi*spf
        f = np.fft.ifft2(g)
        return np.real(f)[1:,1:] # erase the last row and column

    def convolve_scipy(self, im, omega):
        return signal.convolve(im, omega, mode = 'full')

    def convolve_scipy_fft(self, im, omega):
        return signal.fftconvolve(im, omega, mode = 'full')

    def convolve_scipy_2d(self, im, omega):
        return signal.convolve2d(im, omega)

    def convolve_uint_scipy(self, im, omega):
        conv = ndimage.convolve(im, omega, mode = 'constant', cval = 0, origin=0, output=np.int16)
        np.clip(conv,0,255,out=conv)
        return np.uint8(conv)

    #Fastest convolution found for this application, Select this for testing ###############################
    def convolve_uint_scipy_view16(self, im, omega):    
        conv = ndimage.convolve(im, omega, mode = 'constant', cval = 0, origin=0, output=np.int16)
        np.clip(conv,0,255,out=conv)
        return conv.view('uint8')[:,::2]
    ########################################################################################################

    def convolve_uint_scipy_view32(self, im, omega):
        conv = ndimage.convolve(im, omega, mode = 'constant', cval = 0, origin=0, output=np.int32)
        np.clip(conv,0,255,out=conv)
        return conv.view('uint8')[:,::4]
    
    def showFrame(self):
        ret, frame = self.cap.read()
        if ret == True:
            # Converting in shades of gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detecting blur
            time_start = time.time()
            if(self.checked.get() == True): self.setBlurred(self.detectBlur(gray))
            self.frametime_sum_tk += time.time() - time_start
            if(self.n_frame % self.mean_size == 0):
                print("Frametime Blur calculation: {time:.3f}ms".format(time = 1000 * self.frametime_sum_tk/self.mean_size))
                self.frametime_sum_tk = 0

            self.n_frame += 1
            #Setting Test Convolutions tuple and getting kernel
            omega = self.kernel[self.option.get()]
            self.test_convolution = (gray,omega)
            # Blurring the image
            conv = self.convolve(gray, omega, 5)

            if(self.n_frame % self.mean_size == 0):
                new_time = time.time()
                print("Frametime: {time:.3f}ms".format(time = 1000*(new_time - self.last_frametime)/self.mean_size))
                self.last_frametime = new_time

            img = Image.fromarray(conv)
            img = self.imgSizeAdjust(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_frame.imgtk = imgtk
            self.label_frame.configure(image=imgtk)
            self.label_frame.after(5, self.showFrame)

#Set up GUI
window = tk.Tk()
window.title("Convolution of kernels with webcam frames")
main = MyApp(window)
main.showFrame()
#Starts GUI
window.mainloop()
