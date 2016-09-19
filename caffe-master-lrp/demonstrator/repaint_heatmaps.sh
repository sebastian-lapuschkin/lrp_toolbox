cmap=black-firered
hmfolder=./lrp_faces_output/data-and-txt-test/data
synsetfile=/media/lapuschkin/Data/AdienceBechmarkOfUnfilteredFacesForGenderAndAgeClassification/lmdb/faces-227-data-and-txt-test/synset_words.txt

for rawhm in $hmfolder/*_rawhm.txt
do
	python apply_heatmap.py $rawhm $cmap
done

for toptenfile in $hmfolder/*top10scores.txt
do
	python create_readable_scores.py $toptenfile $synsetfile
done

