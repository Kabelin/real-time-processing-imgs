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
    # !pip install opencv-python
    print("opencv-python isn't installed")
if importlib.util.find_spec("PIL") is None:
    # pip3 install Pillow
    print("Pillow isn't installed")
if importlib.util.find_spec("tkinter") is None:
    # sudo apt-get install python3-tk
    print("Tkinter isn't installed")

#Set up GUI
window = tk.Tk()
window.title("Projeto Final: Convolução de kernels sobre a captura do webcam")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
# print(screen_height, screen_width)
fontSize = 14 if screen_width <= 1366 else 16
default_font = tkFont.nametofont("TkDefaultFont")
default_font.configure(size=fontSize)

#Main frames
frame1 = tk.Frame(window, background="#282a36")
frame1.pack(fill="both", expand=True, side="top")
frame2 = tk.Frame(frame1, background="#282a36")
frame2.pack(fill="x", expand=True, side="left", anchor="n")
frame3 = tk.Frame(frame2, background="#282a36")
frame3.pack(fill="x", expand=True, side="top", anchor="n", padx=5)

#----------------Vars----------------#
checked = tk.BooleanVar()
option = tk.StringVar()
blurThresh = tk.DoubleVar()
blurThresh.set(9)

#--------------Widgets--------------#
#Status of blur detection
status = tk.Label(
    frame2, 
    text="Not blurred",
    background="#282a36",
    foreground="#f8f8f2",
)

# Values of blur
blurValue = tk.Label(
    frame2, 
    text="",
    background="#282a36",
    foreground="#f8f8f2",
)

# Threshold adjust label
lblThreshold = tk.Label(
    frame2, 
    text="Threshold adjust:",    
    background="#282a36",
    foreground="#f8f8f2",
)

def showBlurOptions(var = checked):
    if(var.get() == True): 
        status.pack(side="top", pady=(5,0))
        blurValue.pack(side="top")
        lblThreshold.pack(side="top")
        scale.pack(side="top", pady=(0, 5), fill="x", padx=5)
    else: 
        status.pack_forget()
        scale.pack_forget()
        blurValue.pack_forget()
        lblThreshold.pack_forget()

#Chechbutton of blur detection
checkBlur = tk.Checkbutton(
    frame3, 
    text="Blur detection", 
    variable=checked, 
    pady=10,
    padx=5,
    command=showBlurOptions, 
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
checkBlur.pack(side="left")

# Threshold scale
scale = tk.Scale(
    frame2, 
    variable=blurThresh, 
    orient="horizontal",
    background="#282a36",
    activebackground="#44475a",
    foreground="#f8f8f2",
    borderwidth=1,
    from_=-5,
    to=25
)

#Listbox of kernels
listbox = tk.Listbox(
    frame2,
    background="#282a36",
    foreground="#f8f8f2",
    selectforeground="#f8f8f2",
    selectbackground="#44475a",
    borderwidth=1,
    highlightcolor="#f8f8f2",
    relief="solid",
    height=17
)
listbox.pack(side="bottom", pady=5, padx=5, fill="both", expand=True)

#Button for convolution Testing
test_convolution_vars = None
def testLatestConvolution():
    print("Testing convolutions with the last image")
    global test_convolution_vars
    gray, omega = test_convolution_vars
    testConvolutions(getConvolutions(), gray, omega, 1)

testConvolutionButton = tk.Button(
    frame3,
    text= "Test convolutions",
    background="#282a36",
    foreground="#f8f8f2",
    activebackground="#44475a",
    activeforeground="#f8f8f2",
    borderwidth=1,
    pady=9,
    highlightcolor="#f8f8f2",
    relief="solid",
    command = testLatestConvolution
)
testConvolutionButton.pack(side="right", padx=(5,0))

#Capture video frames
lmain = tk.Label(frame1, borderwidth=0)
lmain.pack(side="right", anchor="n")

cap = cv2.VideoCapture(0)

frametimeLast = time.time()
frametimeSum_tk = 0
nFrame = 0
#----------------Functions and kernels----------------#
def show_frame():
    global frametimeLast, frametimeSum_tk, test_convolution_vars, nFrame
    meanSize = 100
    ret, frame = cap.read()
    if ret == True:
      # Converting in shades of gray
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # Detecting blur
      if(checked.get() == True): setBlurred(detectBlur(gray))
      #Setting Test Convolutions tuple and getting kernel
      omega = kernel[option.get()]
      test_convolution_vars = (gray,omega)
      # Blurring the image
      conv = convolve(gray, omega, 5)

      if(nFrame % meanSize == 0):
          newTime = time.time()
          print("Frametime: {time:.3f}ms".format(time = 1000*(newTime - frametimeLast)/meanSize))
          frametimeLast = newTime
     
      timeStart = time.time()

      img = Image.fromarray(conv)      
      imgtk = ImageTk.PhotoImage(image=imgSizeAdjust(img)) 
      lmain.imgtk = imgtk
      lmain.configure(image=imgtk)
      frametimeSum_tk += time.time() - timeStart

      if(nFrame % meanSize == 0):
          print("Frametime tk config: {time:.3f}ms".format(time = 1000 * frametimeSum_tk/meanSize))
          frametimeSum_tk = 0

      nFrame += 1
      lmain.after(10, show_frame) 


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

def setBlurred(tuple): #(value, maxValue)
    # print(tuple)
    isBlurred = tuple[0] <= tuple[1]
    value = '%.2f'%tuple[0]
    maxValue = '%.2f'%tuple[1]
    if(isBlurred):
        status.config(text="Blurred")
        blurValue.config(text="Value: {} of {}".format(value, maxValue))
    else:
        status.config(text="Not blurred")
        blurValue.config(text="Value: {} of {}".format(value, maxValue))


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

# Filling up the listbox
for item in kernelKeys:
    listbox.insert("end", item)

listbox.select_set(0)

# Binding selected kernel of listbox with option variable
def on_selection(event):
    option.set(listbox.get(listbox.curselection()))
 
listbox.bind('<<ListboxSelect>>', on_selection)

# Image resize for bigger resolutions
def imgSizeAdjust(img):
    param = 1.3 if screen_width <= 1366 else 1.6
    return img.resize((int(img.width * param), int(img.height * param)))

#Área de Convoluções e Testes

#Convoluções distintas para teste
def convolve(im, omega, index):
    
    defaultIndex = 1
    convolutions = getConvolutions()
    index = index or defaultIndex
    if(index > len(convolutions)-1):
        print("Convolução com o index {} não encontrado, carregando default".format(index))
        index = defaultIndex
    result = convolutions[index](im,omega)
    
    return result

def testConvolutions(convolutions, im, omega, inverseChance=1):  #1 of inverseChance of executing the test
    if(random.randint(1,inverseChance) == 1):
        for i,convolution in enumerate(convolutions):
            #timeStart = time.time()
            #c = convolution(im,omega)
            #timeEnd = time.time()
            numberOfExecutions = 100
            duration = timeit.timeit(lambda: convolution(im,omega), number=numberOfExecutions)
            print("Tempo de execução da convolução {index}: {duration:.3f}ms".format(index = i ,duration = duration*1000/numberOfExecutions))
            #print(c)


#Implementações das convoluções
def convolve_fft(im,omega):
    im = np.pad(im, ((0,1), (0,1))) # zero pad últimas linha e coluna
    spi = np.fft.fft2(im)
    spf = np.fft.fft2(omega, s=im.shape)
    g = spi*spf
    f = np.fft.ifft2(g)
    return np.real(f)[1:,1:] # elimina as primeiras linha e coluna

def convolve_pure(im,omega):
    M, N = im.shape
    A, B = omega.shape
    a, b = A//2, B//2 # núcleo com dimensões ímpares
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

def convolve_scipy(im, omega):
    return signal.convolve(im, omega, mode = 'full')

def convolve_scipy_fft(im, omega):
    return signal.fftconvolve(im, omega, mode = 'full')

def convolve_scipy_2d(im, omega):
    return signal.convolve2d(im, omega)

def convolve_uint_scipy(im, omega):
    conv = ndimage.convolve(im, omega, mode = 'constant', cval = 0, origin=0, output=np.int16)
    np.clip(conv,0,255,out=conv)
    return np.uint8(conv)

def convolve_uint_scipy_view16(im, omega):    #Fastest convolution yet
    conv = ndimage.convolve(im, omega, mode = 'constant', cval = 0, origin=0, output=np.int16)
    np.clip(conv,0,255,out=conv)
    return conv.view('uint8')[:,::2]

def convolve_uint_scipy_view32(im, omega):
    conv = ndimage.convolve(im, omega, mode = 'constant', cval = 0, origin=0, output=np.int32)
    np.clip(conv,0,255,out=conv)
    return conv.view('uint8')[:,::4]
    

#Armazenamento das convoluções em um array
def getConvolutions():
    convolutions = [
        #convolve_pure,
        convolve_fft,
        convolve_scipy,
        convolve_scipy_fft,
        convolve_scipy_2d,
        convolve_uint_scipy,
        convolve_uint_scipy_view16,
        convolve_uint_scipy_view32
    ]
    return convolutions

show_frame()  #Display
window.mainloop()  #Starts GUI