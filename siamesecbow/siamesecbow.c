#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define max_w 100
const int vocab_hash_size = 30000000;

typedef float real;

struct train_set {
    int *point;
    int len;
};

char read_vocab_file[MAX_STRING], train_file[MAX_STRING], save_file[MAX_STRING];
char *vocab;
real *syn0, *expTable, alpha = 0.025, starting_alpha;
long long vocab_size = 0, vocab_max_size = 50000, layer_size = 200, train_max_size = 1000000, sen_count_actual = 0;
int *train, neg_num = 5, binary = 0, *vocab_hash, iter, *cut, num_threads = 10, sen_num, debug_mode = 2, ptrain = 0;
struct train_set *tset;
clock_t start;
int (*fun)(FILE*);

int init() {
    long long a, b;
    unsigned long long next_random = 1;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer_size * sizeof(real));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer_size;
    }
}
int get_word_hash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}
int add_wordto_vocab(char *word) {
    unsigned long long hash = 0, a;
    for (a = 0; a < strlen(word); a++) vocab[a + vocab_size * max_w] = word[a];
    hash = get_word_hash(word);
    vocab_size ++;
    if (vocab_size + 20 >= vocab_max_size) {
        vocab_max_size += 50000;
        vocab = (char *)realloc(vocab, vocab_max_size * max_w * sizeof(char));
    }
    while (vocab_hash[hash] != -1) hash = (hash + 1) %vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

int search_vocab(char *word) {
    unsigned int hash = get_word_hash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        //if (inx == vocab_hash[hash]) 
        if (!strcmp(word, vocab + vocab_hash[hash] * max_w)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

int read_vocab() {
    printf("loading vocab ...\n");
    long long words, size, a, b, c, d;
    unsigned int hash;
    char word_t[max_w];
    FILE *f;
    f = fopen(read_vocab_file, "rb");
    if (f == NULL) {
        printf("Input vocab file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    vocab_size = words; 
    layer_size = size;
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    a = posix_memalign((void **)&syn0, 128, (long long)words * size * sizeof(float));
    if (syn0 == NULL) {
        printf("Cannot allocate memory\n");
        return -1;
    }
    for (b = 0; b < words; b++) {
        fscanf(f, "%s", word_t);
        for (a = 0; a < strlen(word_t); a++) {
            vocab[b * max_w + a] = word_t[a];
        }
        hash = get_word_hash(vocab + b * max_w);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = b;
        for (c = 0; c < layer_size; c++) {
            fscanf(f, "%f", &syn0[b * layer_size + c]);
        }
    }
    fclose(f);
}

void read_word(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0)  {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else continue;
        }
        word[a] = ch;
        a++;
        if (a >= max_w - 1) a--;
    }
    word[a] = 0;
}

int read_word_index(FILE *fin) {
    char word[max_w];
    read_word(word, fin);
    if (feof(fin)) return -1;
    return search_vocab(word);
}
int get_word_index(FILE *fin) {
    char word[max_w];
    read_word(word, fin);
    if (feof(fin)) return -1;
    int a = search_vocab(word);
    if (a == -1) {
        a = add_wordto_vocab(word);
    }
    return a;
}
int get_train() {
    printf("reading train file ...\n");
    int a, b, c, d, sen[MAX_SENTENCE_LENGTH];
    char word[max_w];
    if (! ptrain) {fun = &get_word_index; add_wordto_vocab("</s>");}
    else fun = &read_word_index;
    FILE *f;
    f = fopen(train_file, "rb");
    if (f == NULL) {
        printf("Input train file not found.\n");
        return -1;
    }
    tset[0].point = (int *)calloc(1, sizeof(int));
    tset[0].point[0] = 0;
    tset[0].len = 1;
    a = 1; c = 0; int e = 0;
    while (1) {
        //b = read_word_index(f);
        b =(*fun)(f);
        //printf("%d %d %d\n", c, b, e);
        if (feof(f)) break;
        if (b == -1) continue;
        if (b == 0 || c + 1 >= MAX_SENTENCE_LENGTH) {
            if (c == 0) continue;
            e ++;
            if (c + 1 >= MAX_SENTENCE_LENGTH) sen[c++] = b;
            tset[a].point = (int *)calloc(c, sizeof(int));
            for (d = 0; d < c; d++) {
                tset[a].point[d] = sen[d];
            }
            tset[a].len = c;
            if (feof(f)) break;
            c = 0;
            a ++;
            if (a + 2 >= train_max_size) {
                train_max_size += 10000;
                tset = (struct train_set*)realloc(tset, train_max_size * sizeof(struct train_set));
            }
            continue;
        }
        sen[c++] = b;
    }
    fclose(f);
    tset[a].point = (int *)calloc(1, sizeof(int));
    tset[a].point[0] = 0;
    tset[a].len = 1;
    train_max_size = a;
    return a;
}

void get_cbow(real *temp, int inx, int inx_) {
    for (int i=0; i < tset[inx].len; i++) {
        for (int j=0; j<layer_size; j++) {
            temp[j + inx_ * layer_size] += syn0[tset[inx].point[i] * layer_size + j];
        } 
    }

    for (int i=0; i<layer_size; i++) {
        temp[i + inx_ * layer_size] /= tset[inx].len;
    }

}
real get_squre(real *temp, int inx) {
    real sum = 0;
    for (int i=0; i<layer_size; i++) {
        sum += temp[i + layer_size * inx] * temp[i + layer_size * inx];
    }
    return sum;
}
real get_cosine(real *temp1, real *temp2, real nm1_, real *nm2_, int inx) {
    real dot = 0;
    for (int i=0; i<layer_size; i++) {
        dot += temp1[i] * temp2[i + layer_size * inx];
    }
    return dot / (nm1_ * nm2_[inx]);
}
void get_softmax(real *temp, real *prop) {
    real sum = 0;
    for (int i=0; i < (neg_num + 2); i++) {
        if (temp[i] >= MAX_EXP) temp[i] = MAX_EXP;
        else if (temp[i] <= -MAX_EXP) temp[i] = -MAX_EXP;
        sum += expTable[(int)((temp[i] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
    }
    for (int i=0; i < (neg_num + 2); i++) {
        prop[i] = expTable[(int)((temp[i] + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))] / sum;
    }
}
void *train_model_thread(void *id) {
    int a, b, c, d, e, id_, rand, local_iter = iter, sen_count = 0, last_sen_count = 0, *neg;
    real ran;

    real *label, *cosine, *grad1, *grad2, *prop, nm1, nm1_, *nm2, *nm2_;
    label = (real *)calloc((neg_num + 2), sizeof(real));
    for (a = 0; a < neg_num + 2; a++) {
        label[a] = 0;
    }
    label[0] = 0.5; label[1] = 0.5;
    cosine = (real *)calloc((neg_num + 2), sizeof(real));
    grad1   = (real *)calloc(layer_size , sizeof(real));
    grad2   = (real *)calloc(layer_size * (neg_num + 2) , sizeof(real));
    prop   = (real *)calloc((neg_num + 2), sizeof(real));
    nm2    = (real *)calloc((neg_num + 2), sizeof(real));
    nm2_    = (real *)calloc((neg_num + 2), sizeof(real));
    neg    = (int *)calloc((neg_num + 2), sizeof(int));

    real *neu1 = (real *)calloc(layer_size, sizeof(real));
    real *neu2 = (real *)calloc((neg_num + 2) * layer_size, sizeof(real));
    clock_t now;
    unsigned long long next_random = (long long)id;
    id_ = (intptr_t)id; 
    while (local_iter-- > 0) {
        a = cut[id_];
        while (a < cut[id_ + 1]){
            if (sen_count - last_sen_count > 100) {
                sen_count_actual += sen_count - last_sen_count;
                last_sen_count = sen_count;
                if (debug_mode > 1) {
                    now = clock();
                    printf("%cAlpha: %f Progress: %.2f%% Sentences/thread/sec: %.2fk  ", 13, alpha,
                        sen_count_actual / (real)(iter * sen_num + 1) * 100,
                        sen_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                    fflush(stdout);
                }
                alpha = starting_alpha * (1 - sen_count_actual / (real)(iter * sen_num + 1));
                if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
            }
            b = 0;
            for (b = 0; b < layer_size; b++) neu1[b] = 0;
            for (b = 0; b < (neg_num + 2) * layer_size; b++) neu2[b] = 0;
            get_cbow(neu1, a, 0);
            nm1 = get_squre(neu1, 0);
            nm1_ = sqrt(nm1);
            neg[0] = a - 1; neg[1] = a + 1;
            b = 0;
            while (b < neg_num) {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                ran = (next_random & 0xFFFF) / (real)65536;
                rand = (int)(ran * train_max_size);
                neg[b+2] = rand; b++;
            }
            b = 0;
            while (b < neg_num + 2) {
                get_cbow(neu2, neg[b], b);
                nm2[b] = get_squre(neu2, b);
                nm2_[b] = sqrt(nm2[b]);
                b++;
            } 
            b = 0;
            while (b < neg_num + 2) {
                cosine[b] = get_cosine(neu1, neu2, nm1_, nm2_, b);
                b ++;
            }
            get_softmax(cosine, prop);
            b = 0;
            for (c = 0; c < layer_size; c++) grad1[c] = 0.0;
            //get the gradients
            while (b < neg_num + 2) {    
                for (c = 0; c < layer_size; c++) {
                    grad2[c + layer_size * b] = (label[b] - prop[b]) * (cosine[b] / nm2[b] *neu2[c + layer_size * b] + neu1[c] / (nm1_ * nm2_[b]));
                    grad1[c] += 1.0 / (neg_num + 2) * (label[b] - prop[b]) * (cosine[b] / nm1 * neu1[c] + neu2[c + layer_size * b] / (nm1_ * nm2_[b]));
                }
                b++;
            }
            b = 0; d = tset[a].len;
            //update the parameters
            for (b = 0; b < d; b++) {
                for (c = 0; c < layer_size; c++) {
                    syn0[c + tset[a].point[b] * layer_size] -= alpha * (1.0 / d) * grad1[c];
                }
            }
            b = 0; d = 0;
            while (b < neg_num + 2) {
                e = tset[neg[b]].len;
                for (c = 0; c < e; c++) {
                    for (d = 0; d < layer_size; d++) {
                        syn0[d + tset[neg[b]].point[c] * layer_size] -= alpha * (1.0 / e) * grad2[d + layer_size * b];
                    }
                }
                b++;
            }
            a ++; sen_count ++;
        }
    }
}

void train_model() {
    printf("train model ...\n");
    int a, b, c, d = 0; long e;
    FILE *fo;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    starting_alpha = alpha;
    start = clock();
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    if (ptrain) { printf("use pre-trained vocabulary\n");read_vocab(); d = get_train(); }
    else { 
        printf("Don't use pre-trained vocabulary\n"); 
        vocab = (char *)malloc((long long)vocab_max_size * max_w * sizeof(char));
        d = get_train(); 
        init();
    }
    //d = get_train();
    sen_num = d;
    cut = (int *)calloc((num_threads + 1), sizeof(pthread_t));
    b = d / num_threads;
    c = d % num_threads;
    cut[0] = 1;
    for (a = 1; a < num_threads; a++){
        cut[a] = a * b;
        printf(" %d ", cut[a]);
    }printf("%d \n", d);
    cut[num_threads] = d;
    printf("train model thread ...\n");
    for (e = 0; e < num_threads; e++) pthread_create(&pt[e], NULL, train_model_thread, (void *)e);
    for (e = 0; e < num_threads; e++) pthread_join(pt[e], NULL);
    printf("\nsave vocab ...\n");
    fo = fopen(save_file, "wb");
    fprintf(fo, "%lld %lld\n", vocab_size, layer_size);
    for (a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab + a * max_w);
        if (binary) for (b = 0; b < layer_size; b++) fwrite(&syn0[a * layer_size + b], sizeof(real), 1, fo);
        else for (b = 0; b < layer_size; b++) fprintf(fo, "%lf ", syn0[a *layer_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);
}

int arg_pos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}
int main (int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("siamese cbow toolkit v 1\n");    
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe saved vocabulary file");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.05.\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug model\n");
        printf("\t-iter <int>\n");
        printf("\t\tThe iteration times of training\n");
        printf("\t-ptrain <int>\n");
        printf("\t\tIf use pre-trained vocabulary, set the value 1 and enbale -read-vocab, or 0\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe pre-trained vocabulary path\n");
        return 0;
    }
    read_vocab_file[0] = 0;
    train_file[0] = 0;
    save_file[0] = 0;
    if ((i = arg_pos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = arg_pos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_file, argv[i + 1]);
    if ((i = arg_pos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = arg_pos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = arg_pos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = arg_pos((char *)"-ptrain", argc, argv)) > 0) ptrain = atoi(argv[i + 1]);

    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
        expTable[i] = expTable[i] / (expTable[i] + 1);
    }
    tset = (struct train_set*)calloc(train_max_size, sizeof(struct train_set));
    train_model();
    return 0;
}
