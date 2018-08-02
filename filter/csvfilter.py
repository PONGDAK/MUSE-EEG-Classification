import csv

FILE_COUNT = 5
FILE_NAME = 'sleepness_'
ROW_COUNT = 15360   # 128 * x
DATA_TYPE = ' Person0/eeg'

time = []
eeg1 = []
eeg2 = []
eeg3 = []
eeg4 = []

for i in range(1, FILE_COUNT + 1):
    filename = FILE_NAME + str(i) + ".csv"
    with open('./data/' + filename, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile, delimiter=',')
        count = 0
        for idx, row in enumerate(reader):
            if row[1] == DATA_TYPE:
                count += 1
                if count <= ROW_COUNT :  #and count % 4 == 0
                    time.append(float(row[0]))
                    eeg1.append(float(row[2]))
                    eeg2.append(float(row[3]))
                    eeg3.append(float(row[4]))
                    eeg4.append(float(row[5]))

with open('./filtered/filtered_' + FILE_NAME + '.csv', 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    for i in range(0, int(FILE_COUNT * ROW_COUNT)):
        writer.writerow([eeg1[i], eeg2[i], eeg3[i], eeg4[i]])
