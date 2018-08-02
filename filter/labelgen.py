import csv
SEQ_LEN = 128
DATA_COUNT = 92160

with open('label_test.csv', 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    for j in range(1, 7):
        for i in range(0, int(DATA_COUNT/SEQ_LEN/6)):
            writer.writerow([j])
