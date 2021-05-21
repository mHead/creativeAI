import pandas as pd
import os
try:
	imagesPath = '/Volumes/SamsungSSD/creativeAI/imageSide/dataset_stringRevisioned/wikiart/Ukiyo_e/'
	savePath = '/Volumes/SamsungSSD/creativeAI/imageSide/dataset_stringRevisioned/wikiart_v1.2.csv'
	startPoint = '/Volumes/SamsungSSD/creativeAI/imageSide/dataset_stringRevisioned/start.csv'

	df = pd.read_csv(savePath)
	df = df.drop(['Unnamed: 0'], axis=1)
	cnt = 0
	tot = len(os.listdir(imagesPath))
	data = []

	#for imgPath in os.listdir(imagesPath):
	#	save_path = imgPath
	#	data.append([save_path])
	#	cnt += 1
	#	print("Record added!")

	#data = pd.DataFrame(data, columns=['painting'])
	#df = df.append(data, ignore_index=True)
	df.to_csv(savePath)



except FileNotFoundError:
	print("provided image path is not founded")