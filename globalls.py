import os
import glob
import csv

# where global variables go
CR_DEFAULT_TRAVEL_HOP = 0.05

################################## OTHER FUNCTIONS

def clear_and_make_directory(dirPath):

    if not os.path.isdir(dirPath):
        # make directory
        os.mkdir(dirPath)
    else:
        # TODO: clear directory contents
        files = glob.glob(dirPath + "/*")
        for f in files:
            os.remove(f)

def make_csv_file(filePath, columnLabels):
    with open(filePath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columnLabels)
