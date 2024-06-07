import pandas as pd
import sys
import os

database_path_pref = "/mnt/Y/DZ/"
input_path_suff = "/file_based/record_centers.txt"
output_path_suff = "/file_based/reimaged_data.txt"

if __name__=='__main__':

	if len(sys.argv) != 2:
		sys.exit(f"requires slide ID as the only input")
	else:
		slide_id = sys.argv[1]
		input_path  = database_path_pref + slide_id + input_path_suff
		output_path = database_path_pref + slide_id + output_path_suff

	if not os.path.isfile(input_path):
		sys.exit(f"Cannot find input file in {input_path}!")
	
	with open(input_path, 'r') as file:
		lines = file.readlines()

	# Create empty lists for variable names and values
	var_id = []
	var_names = []
	var_values = []

	# Loop through each line in the file
	for i, line in enumerate(lines):
		# Split the line by '&' character
		vars = line.strip().strip('"').split('&')
		# Loop through each variable in the line
		for var in vars:
			# Split the variable by '=' character to get the name and value
			name, value = var.split('=')
			# Append the name and value to their respective lists
			var_id.append(i)
			var_names.append(name)
			var_values.append(value)

	# Create a pandas data frame from the lists
	df = pd.DataFrame({'index':var_id ,'var': var_names, 'val': var_values})
	df = df.pivot(index='index', columns='var', values='val')
	df.drop_duplicates(inplace=True, ignore_index=True, keep='last')
	df.to_csv(output_path, sep='\t', index=False)
