import os
import csv
import pandas as pd
import numpy as np

def sanitize_string(str_to_sanitize):
  #bad_chars = [' ', 'ã', 'â', 'â', 'å','ā', '\xa0' ]

  str_to_sanitize = str_to_sanitize.replace(' ', '')
  str_to_sanitize = str_to_sanitize.replace('ã', 'a')
  str_to_sanitize = str_to_sanitize.replace('â ', 'a') 
  str_to_sanitize = str_to_sanitize.replace('â', 'a')
  str_to_sanitize = str_to_sanitize.replace('å', 'a')
  str_to_sanitize = str_to_sanitize.replace('ā', 'a')
  str_to_sanitize = str_to_sanitize.replace('â\xa0', 'a')

  str_to_sanitize = str_to_sanitize.replace('ë', 'e')
  str_to_sanitize = str_to_sanitize.replace('ê', 'e')
  str_to_sanitize = str_to_sanitize.replace('ė', 'e')
  str_to_sanitize = str_to_sanitize.replace('ē', 'e')


  str_to_sanitize = str_to_sanitize.replace('ï', 'i')
  str_to_sanitize = str_to_sanitize.replace('î', 'i')
  str_to_sanitize = str_to_sanitize.replace('į', 'i')
  str_to_sanitize = str_to_sanitize.replace('ī', 'i')

  str_to_sanitize = str_to_sanitize.replace('ö', 'o')
  str_to_sanitize = str_to_sanitize.replace('ô', 'o')
  str_to_sanitize = str_to_sanitize.replace('ø', 'o')
  str_to_sanitize = str_to_sanitize.replace('õ', 'o')
  str_to_sanitize = str_to_sanitize.replace('ō', 'o')

  str_to_sanitize = str_to_sanitize.replace('©', '')
  str_to_sanitize = str_to_sanitize.replace('³', '')
  str_to_sanitize = str_to_sanitize.replace('¼', '')
  str_to_sanitize = str_to_sanitize.replace('¶', '')


  str_to_sanitize = str_to_sanitize.replace('ü', 'u')
  str_to_sanitize = str_to_sanitize.replace('û', 'u')
  str_to_sanitize = str_to_sanitize.replace('ū', 'u')

  str_to_sanitize = str_to_sanitize.replace('¨', '')
  str_to_sanitize = str_to_sanitize.replace('lacombeâ ', 'lacombea')
  str_to_sanitize = str_to_sanitize.replace('a\xa0', 'a')

  return str_to_sanitize



def print_filtered_rows(dataFrame, string_to_search='lacomb'):
	for row in dataFrame.itertuples():
		if row[2].__contains__(string_to_search):
			print("OLD: {}".format(row[2]))
			newname = sanitize_string(row[2])
			#row[2] = newname
			print("NEW: {}".format(newname))


def sanitize_csv(dataFrame):
	data = []
	nSanitizes = 0
	flag = 0
	for row in dataFrame.itertuples():
		newname = sanitize_string(row[2])
		if(newname != row[2]):
			print("OLD: {}".format(row[2]))
			print("NEW: {}".format(newname))
			nSanitizes += 1

		#data.append([style, sanitized_name, dist.tolist()])





artemis_release_v0 = '/Volumes/SamsungSSD/creativeAI/imageSide/official_data/artemis_dataset_release_v0.csv'

df = pd.read_csv(artemis_release_v0)
print(len(df))
print(df.columns)
print(type(df))

print_filtered_rows(df)




