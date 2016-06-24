
#gcc cwe2.c -g -o cwe2 -lm -pthread -O0 -march=native -Wall -funroll-loops -Wno-unused-result
time ./cwe2 -train /home/yuanye/lab/word2vec-master/data/wiki_sogo_cut -output-word wiki_sogo_cwe2100 -output-char wiki_charcb2 -save-pos-file wiki_poscb2 -cbow 1 -size 100 -window 8 -sample 1e-4 -negative 25 -hs 0 -iter 30 -threads 10 -min-count 1
