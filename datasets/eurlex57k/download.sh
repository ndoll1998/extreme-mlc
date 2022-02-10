wget -O ./datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip ./datasets.zip -d ./
mv dataset/* ./
rm -r dataset
rm ./datasets.zip
rm -rf ./__MACOSX
wget -O EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
