import csv
files_name = ['output_03291913.csv', 'output_03271133.csv', 'output_03291103.csv']
file_readers = [open(file_name, 'r') for file_name in files_name]
out_file_name = 'ensemble.csv'
out_file = open(out_file_name, 'w')
writer = csv.writer(out_file)
header = ['Id', 'Category']
writer.writerow(header)
for reader in file_readers:
    next(reader)
count = {}
for reader in file_readers:
    for row in reader:
        row = list(row.replace('\n', '').split(','))
        if row[0] in count.keys():
            if row[1] in count[row[0]].keys():
                count[row[0]][row[1]] += 1
            else:
                count[row[0]][row[1]] = 1
        else:
            count[row[0]] = {row[1]:1}
for idx in list(count.keys()):
    mx_idx = list(count[idx].keys())[0]
    for _idx in list(count[idx].keys())[1:]:
        if count[idx][mx_idx] < count[idx][_idx]:
            mx_idx = _idx
    writer.writerow([idx, mx_idx])
