import csv

txt_file = r"icuadmit.txt"
csv_file = r"icuadmit.csv"
txt2_file = r"collection.txt"
csv2_file = r"collection.csv"


in_txt = csv.reader(open(txt_file, "rb"), delimiter = '\t')
out_csv = csv.writer(open(csv_file, 'wb'))

out_csv.writerows(in_txt)


in_txt2 = csv.reader(open(txt2_file, "rb"), delimiter = '\t')
out_csv2 = csv.writer(open(csv2_file, 'wb'))

out_csv2.writerows(in_txt2)
