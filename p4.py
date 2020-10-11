import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import importlib
if importlib.util.find_spec("PIL") is None:
    # pip3 install Pillow
    print("Pillow isn't installed")
if importlib.util.find_spec("tkinter") is None:
    # sudo apt-get install python3-tk
    print("Tkinter isn't installed")

#Set up GUI
window = tk.Tk()
window.title("Projeto Final: Convolução de kernels sobre a captura do webcam")

#Main frames
frame1 = tk.Frame(window, background="#282a36")
frame1.pack(fill="both", expand=True, side="top")
frame2 = tk.Frame(frame1, background="#282a36")
frame2.pack(fill="x", expand=True, side="left", anchor="n")

#Vars
var = tk.BooleanVar()
option = tk.StringVar()

#Status bar for blur detection
status = tk.Label(frame2, text="Not blurred", pady=10)
blurValue = tk.Label(frame2, text="")
lblThreshold = tk.Label(frame2, text="Threshold adjust:")

def setBlurred(tuple): #(value, maxValue)
    print(tuple)
    isBlurred = tuple[0] <= tuple[1]
    value = '%.2f'%tuple[0]
    maxValue = '%.2f'%tuple[1]
    if(isBlurred):
        status.config(text="Blurred")
        blurValue.config(text="Value: {} of {}".format(value, maxValue))
    else:
        status.config(text="Not blurred")
        blurValue.config(text="Value: {} of {}".format(value, maxValue))

def showBlurOptions(var = var):
    if(var.get() == True): 
        status.pack(side="top")
        blurValue.pack(side="top", pady=(0, 10))
        lblThreshold.pack(side="top")
        scale.pack(side="top", pady=(0, 10))
    else: 
        status.pack_forget()
        scale.pack_forget()
        blurValue.pack_forget()
        lblThreshold.pack_forget()

#Blur detection
checkBlur = tk.Checkbutton(
                            frame2, 
                            text="Blur detection", 
                            width=20, variable=var, 
                            pady=10, 
                            command=showBlurOptions, 
                            background="#282a36",
                            highlightbackground="#00b9ff",
                            activebackground="#44475a",
                            activeforeground="#f8f8f2",
                            foreground="#f8f8f2",
                            selectcolor="#44475a"
                          )
checkBlur.pack(side="top", fill="x")

#Initial value for Blur Treshold To decide when its blurred or not
blurThresh = tk.DoubleVar()
blurThresh.set(9)
# def increaseBlurTresh():
#     global blurThresh
#     blurThresh += 1
# def decreaseBlurTresh():
#     global blurThresh
#     blurThresh -= 1
# #Buttons To increase blurThresh Value
# ButtonFrames = tk.Frame(window, width=200)
# ButtonFrames.grid(row = 0, column = 3)
# ButtonPlus = tk.Button(ButtonFrames, text="+", command = increaseBlurTresh)
# ButtonPlus.grid(row = 0, column = 0)
# ButtonMinus = tk.Button(ButtonFrames, text="-", command = decreaseBlurTresh)
# ButtonMinus.grid(row = 1, column = 0)

scale = tk.Scale(frame2, variable=blurThresh, orient="horizontal")


kernel = {
    'identity':                 np.array([[0,0,0],[0,1,0],[0,0,0]],                                                         dtype=float),
    'edge detection':           np.array([[1,0,-1],[0,0,0],[-1,0,1]],                                                       dtype=float),
    'laplacian':                np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],                                                     dtype=float),
    'laplacian w/ diagonals':   np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],                                                 dtype=float),
    'laplacian of gaussian':    np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]],        dtype=float),
    'scharr':                   np.array([[-3, 0, 3],[-10,0,10],[-3, 0, 3]],                                                dtype=float),
    'sobel edge horizontal':    np.array([[-1,-2,-1],[0,0,0],[1,2,1]],                                                      dtype=float),
    'sobel edge vertical':      np.array([[-1,0,1],[-2,0,2],[-1,0,1]],                                                      dtype=float),
    'line detection horizontal':np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]],                                                   dtype=float),
    'line detection vertical':  np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]],                                                   dtype=float),
    'line detection 45°':       np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]],                                                   dtype=float),
    'line detection 135°':      np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]],                                                   dtype=float),
    'box blur':                 (1/9)*np.ones((3,3),                                                                        dtype=float),
    'gaussian blur 3x3':        (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]],                                                  dtype=float),
    'gaussian blur 5x5':        (1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],    dtype=float),
    'sharpen':                  np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],                                                     dtype=float),
    'unsharp masking':          (-1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
}

#Getting kernel keys and setting first option
kernelKeys = kernel.keys()
option.set(list(kernelKeys)[0])

#Listbox component
def on_selection(event):
    # print('(event) previous:\t{}'.format(event.widget.get('active')))
    # print('(event) current:\t{}'.format(event.widget.get(event.widget.curselection())))
    option.set(listbox.get(listbox.curselection()))

listbox = tk.Listbox(frame2)
listbox.pack(side="bottom", pady=10, fill="x")

for item in kernelKeys:
    listbox.insert("end", item)

listbox.bind('<<ListboxSelect>>', on_selection)

#Capture video frames
lmain = tk.Label(frame1, borderwidth=0)
lmain.pack(side="right")

cap = cv2.VideoCapture(0)
def show_frame():
    ret, frame = cap.read()
    if ret == True:
      # Converting in shades of gray
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Detecting blur
      if(var.get() == True): setBlurred(detectBlur(gray))
      # Blurring the image
      conv = np.uint8(np.round(convolve(gray, kernel[option.get()], fft=True)))
      img = Image.fromarray(conv)
      imgtk = ImageTk.PhotoImage(image=img)
      lmain.imgtk = imgtk
      lmain.configure(image=imgtk)
      lmain.after(10, show_frame) 

def convolve(im, omega, fft=False):
    M, N = im.shape
    A, B = omega.shape
    a, b = A//2, B//2 # núcleo com dimensões ímpares
    if not fft:
        f = np.array(im, dtype=float)
        g = np.zeros_like(f, dtype=float)
        for x in range(M):
            for y in range(N):
                aux = 0.0
                for dx in range(-a, a+1):
                    for dy in range(-b, b+1):
                        if 0 <= x+dx < M and 0 <= y+dy < N: # ou você pode usar "zero pad" na imagem
                            aux += omega[a-dx, b-dy]*f[x+dx, y+dy]
                g[x, y] = aux
        return g
    else:
        im = np.pad(im, ((0,1), (0,1))) # zero pad últimas linha e coluna
        spi = np.fft.fft2(im)
        spf = np.fft.fft2(omega, s=im.shape)
        g = spi*spf
        f = np.fft.ifft2(g)
        return np.real(f)[1:,1:] # elimina as primeiras linha e coluna

def detectBlur(im,cutFreq=60,thresh=None):
    thresh = thresh or blurThresh.get()
    fft = np.fft.fft2(im)
    (h, w) = fft.shape
    fft[0:cutFreq,0:cutFreq]     = 0
    fft[h-cutFreq:h,0:cutFreq]   = 0
    fft[0:cutFreq,w-cutFreq:w]   = 0 
    fft[h-cutFreq:h,w-cutFreq:w] = 0
    recon = np.fft.ifft2(fft)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return (mean, thresh)

show_frame()  #Display
window.mainloop()  #Starts GUI
