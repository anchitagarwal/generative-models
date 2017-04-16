import imageio
from PIL import Image
import glob
imageio.plugins.ffmpeg.download()
import moviepy.editor as mpy
import numpy as np

def make_gif(images, fname, duration=2, true_image=False):
	def make_frame(t):
		try:
			x = images[int(len(images)/duration*t)]
		except:
			x = images[-1]

		if true_image:
			return x.astype(np.uint8)
		else:
			return ((x+1)/2*255).astype(np.uint8)

	clip = mpy.VideoClip(make_frame, duration=duration)
	clip.write_gif(fname, fps = len(images) / duration)

if __name__ == '__main__':
    image_dir = 'images_gif/*.jpg'
    image_list = []
    for filename in glob.glob(image_dir):
        im = np.array(Image.open(filename))
        image_list.append(im)
    make_gif(image_list, '../assets/celeb.gif', duration=8, true_image=True)