import datetime

from .interpolations import get_interpolations

a = datetime.datetime.now()

torch_images_list = get_interpolations('images/im1.png', 'images/im3.png', 49)
print(torch_images_list.size())

b = datetime.datetime.now()
print(b-a)
#print(torch_images_list)