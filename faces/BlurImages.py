import cv2
'version opencv-python =  4.2.0.32'
import numpy as np
'numpy.version.version = 1.18.2  (en Python console)'
import os
import glob



#path = r'C:/Users/charl/Documents/1CarpetaPersonal/Fi_UNAM/IFC/Tesis/Imágenes/training_cats_blur'
path = os.getcwd()
# Busqueda de archivos por extension (.jpg o .png)
imagenes = glob.glob(path + os.sep + '*.png')+glob.glob(path + os.sep + '*.jpg') # Lista de todos los archivos con extension .jpg o .png en la carpeta (imagenes) (ruta entera)
#imagenes = glob.glob(path + '*.png')+glob.glob(path + '*.jpg') # Lista de todos los archivos con extension .jpg o .png en la carpeta (imagenes) (ruta entera)
print(len(imagenes))
#destination = r'C:/Users/charl/Documents/1CarpetaPersonal/Fi_UNAM/IFC/Tesis/Imágenes/training_cats_blur'
destination = os.getcwd()

def blur(input_image):
    grade=15
    blur = cv2.blur(input_image, (grade, grade))
    return blur

def posterize(input_image):
    n = 3  # Number of levels of quantization

    indices = np.arange(0, 256)  # List of all colors

    divider = np.linspace(0, 255, n + 1)[1]  # we get a divider

    quantiz = np.int0(np.linspace(0, 255, n))  # we get quantization colors

    color_levels = np.clip(np.int0(indices / divider), 0, n - 1)  # color levels 0,1,2..

    palette = quantiz[color_levels]  # Creating the palette

    im2 = palette[input_image]  # Applying palette on image

    im2 = cv2.convertScaleAbs(im2)  # Converting image back to uint8

    return im2


def gaussian_blur(input_image):
    ksize=15
    ksize_width = ksize
    ksize_height = ksize
    # Both must be positive and odd (impar).

    gausBlur = cv2.GaussianBlur(input_image, (ksize, ksize), sigmaX=0)

    return gausBlur


def saturate(input_image):
    # Define mask's boundaries
    color1 = (28, 145, 0)
    color2 = (82, 255, 252)  # Verde

    hsv = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)

    lower = np.array([color1[0], color1[1], color1[2]])
    upper = np.array([color2[0], color2[1], color2[2]])

    mask = cv2.inRange(hsv, lower, upper)

    background = np.full(input_image.shape, 255, dtype=np.uint8)
    green2black = cv2.bitwise_not(background, input_image, mask=mask)

    return green2black
	
def make_dir():
	import errno
	try:
		os.mkdir('Images_blur')
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


def save(b=None, p=None, gb=None, s=None):
	for i, img in enumerate(imagenes):
		# print (img) Rutas de todas las imagenes
		# print(img.rsplit(os.sep, 1)[1]) Nombre de todas las imagenes
		filename = img.rsplit(os.sep, 1)[1]
		edit = cv2.imread(filename)

		'ACTIONS'
		if b:
			edit = blur(edit)
		if p:
			edit = posterize(edit)
		if gb:
			edit = gaussian_blur(edit)
		if s:
			edit = saturate(edit)
		make_dir()
		cv2.imwrite('Images_blur/'+filename, edit)
		print(i+1,'de',np.size(imagenes))

if __name__ == '__main__':
    save(b=True, p=True, gb=True, s=True)