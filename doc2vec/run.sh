
time ./doc2vec -train /home/renyafeng/code/data/tt20_newsg/train -output word_vec_newsg_1 -output-para-file doc_vec_newsg_cbow_1 -cbow 1 -size 200 -window 15 -negative 5 -hs 0 -sample 1e-5 -threads 20 -binary 0 -iter 20 min-count 5 -pretrain 1 -pretrain-file /home/renyafeng/embedding/wikipedia/vector_200.nn -pretrain-vocab-size 400000

#time ./doc2vec -train /home/renyafeng/code/data/cornell_movie_dialogs_corpus/cornell_movie_dialogs_corpus/movie_lines_2 -output word_vec_newsg -output-para-file doc_vec_newsg -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 8 -binary 0 -iter 15 min-count 1 
