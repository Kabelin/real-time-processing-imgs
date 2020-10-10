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
window.title("Testing processing")
window.config(background="#EEE")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=600)
imageFrame.grid(row=1, column=0, padx=10, pady=10)

#Options frame
optFrame = tk.Frame(window, width=600)
optFrame.grid(row=0, column=0, padx=10, pady=10)

#Vars
var = tk.BooleanVar()
option = tk.StringVar()
option.set("identity")

#Status bar for blur detection
status = tk.Label(optFrame, text="Not blurred", padx=10)


def show():
    if(var.get() == True): 
        status.config(text="Not blurred")
        status.grid(row=0, column=2)
    else: 
        status.config(text="Blurred")
        status.grid_remove()
        

#Blur detection
checkBlur = tk.Checkbutton(optFrame, text="Blur detection", variable=var, padx=10, command=show)
checkBlur.grid(row=0, column=1, padx=40)

#Dropdown menu
drop = tk.OptionMenu(optFrame, option, 
    'identity',
    'edge detection',
    'laplacian',
    'laplacian w/ diagonals',
    'laplacian of gaussian',
    'scharr',
    'sobel edge horizontal',
    'sobel edge vertical',
    'line detection horizontal',
    'line detection vertical',
    'line detection 45°',
    'line detection 135°',
    'box blur',
    'gaussian blur 3x3',
    'gaussian blur 5x5',
    'sharpen',
    'unsharp masking')
drop.grid(row=0, column=0)

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=1, column=0)

cap = cv2.VideoCapture(0)
def show_frame():
    ret, frame = cap.read()
    if ret == True:
      # Converting in shades of gray
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Blurring the image
      conv = np.uint8(np.round(convolve(gray, kernel['laplacian of gaussian'], fft=True)))
      img = Image.fromarray(conv)
      imgtk = ImageTk.PhotoImage(image=img)
      lmain.imgtk = imgtk
      lmain.configure(image=imgtk)
      lmain.after(20, show_frame) 

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

kernel = {
    'identity': np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=float),
    'edge detection': np.array([[1,0,-1],[0,0,0],[-1,0,1]], dtype=float),
    'laplacian': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float),
    'laplacian w/ diagonals': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float),
    'laplacian of gaussian': np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]], dtype=float),
    'scharr': np.array([[-3, 0, 3],[-10,0,10],[-3, 0, 3]], dtype=float),
    'sobel edge horizontal': np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float),
    'sobel edge vertical': np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float),
    'line detection horizontal': np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]], dtype=float),
    'line detection vertical': np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], dtype=float),
    'line detection 45°': np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]], dtype=float),
    'line detection 135°': np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], dtype=float),
    'box blur': (1/9)*np.ones((3,3), dtype=float),
    'gaussian blur 3x3': (1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=float),
    'gaussian blur 5x5': (1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
    'sharpen': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
    'unsharp masking': (-1/256)*np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,-476,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=float),
}

show_frame()  #Display
window.mainloop()  #Starts GUI
