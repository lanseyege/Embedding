gcc -o siamesecbow_ siamesecbow.c -pthread -lm

time ./siamesecbow_ -train f -read-vocab /code/data/CorpusACL-master/sen/cbow_min_5  -save-vocab ss_cbow_min_5 -alpha 0.05 -iter 10 -threads 10 -ptrain 1
