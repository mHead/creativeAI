import os

wikiart_img_dir = '/Volumes/SamsungSSD/creativeAI/imageSide/dataset_stringRevisioned/wikiart'
pathiter = (os.path.join(root, filename) for root, _, filenames in os.walk(wikiart_img_dir) for filename in filenames)
for path in pathiter:
	if path.__contains__('Impressionism/joaqua­n-sorolla'):
		newname = path.replace('joaqua­n-sorolla', 'joaqua­n-sorolla')
		print("OLD:", path)
		print("NEW:", newname)
		if newname != path:
		  os.rename(path, newname)