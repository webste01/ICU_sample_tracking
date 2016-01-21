mysql -B -u pathogendb_rw -h db.hpc.mssm.edu -D vanbah01_pathogens -pitedZit0 -e "SELECT eRAP_ID, admission_datetime, room, report_date FROM tIcuAdmissions;" > icuadmit.txt
mysql -B -u pathogendb_rw -h db.hpc.mssm.edu -D vanbah01_pathogens -pitedZit0 -e "SELECT eRAP_ID, collection_location, collection_date, sampling_date,freezer_ID FROM tStoolCollection;" > collection.txt

sed 's/[fF]zB-//g' collection.txt > tmp
mv tmp collection.txt


#Formats the sql query results into python readable files, one for each ICU
#Takes in: collection.txt and icuadmit.txt
#Outputs: [ICU]_collection.csv and [ICU]_admit.csv

awk '{print $1,$2,$4,$5}' icuadmit.txt > tmp.txt
python convert_txt_csv.py 
