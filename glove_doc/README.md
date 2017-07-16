# glove doc
Firstly, we treate every document as a "unique word"; Then, we count the concurrence of the "unique word"(document) and words in it; Finally, we get the document embedding by training the Glove model.

For the code, glove.c, shuffle.c and vocab\_count.c keep same, we just change cooccur.c to cooc2.c
