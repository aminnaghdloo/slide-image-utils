import pandas as pd
import sys


def hdf2txt(hdf_file, txt_file):
	"""Convert hdf5 file to txt file"""
	df = pd.read_hdf(hdf_file, mode='r', key='features')
	df = df[df.pred == 1]
	df['slide_id'] = hdf_file.split('/')[-1].split('.')[0]
	df.to_csv(txt_file, sep='\t', index=False)


def main():
	if len(sys.argv) != 3:
		print("Usage: python extract_hdf_features.py hdf_file txt_file")
		sys.exit(0)
	else:
		hdf_file = sys.argv[1]
		txt_file = sys.argv[2]
		hdf2txt(hdf_file, txt_file)
		print(f"Successfully extracted features form {sys.argv[1]} to {sys.argv[2]}!")


if __name__ == '__main__':
	main()
