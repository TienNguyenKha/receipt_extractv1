import csv

with open('output.tsv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['hehe Nguyễn'])
    writer.writerow(['hehe Nguyễn'])