#./vocab_count -verbose 2 -max-vocab 200000 -min-count 10 < /home/renyafeng/code/data/CorpusACL-master/natt/aset > ./res/vocab_na1.txt
./cooc2 -verbose 2 -symmetric 0 -window-size 10 -vocab-file ./res/vocab_na1.txt -memory 8.0 -overflow-file tempoverflow < /home/renyafeng/code/data/CorpusACL-master/natt/aset > ./res/coo_na1.bin
./shuffle -verbose 2 -memory 8.0 < ./res/coo_na1.bin > ./res/coo_na1.shuf.bin
./glove -input-file ./res/coo_na1.shuf.bin -vocab-file ./res/vocab_na1.txt -save-file ./res/vectors_na1_100 -corpus-name /home/renyafeng/code/data/CorpusACL-master/natt/aset -gradsq-file ./res/gradsq -verbose 2 -vector-size 100 -threads 8 -alpha 0.75 -x-max 100.0 -eta 0.05 -binary 2 -model 2a

line=$(head -n 4 `basename "$0"`)
echo $line | cat - record > temp_t && mv temp_t record

