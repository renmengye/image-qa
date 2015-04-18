#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

#define max_size 2000         // max length of strings
#define N 40                  // number of closest words that will be shown
#define max_w 50              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char file_name[max_size];
  long long words, size, a, b, c, d;
  long long cursor = 0;
  char *vocab;
  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  cursor += 2 * sizeof(long long);
  FILE *vocab_file;
  vocab_file = fopen("vocab_index.txt", "w+");
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));

  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {
	  vocab[b * max_w + a] = fgetc(f);
	  cursor += sizeof(char);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
	  if ((a < max_w) && (vocab[b * max_w + a] != '\n')) {
		  fprintf(vocab_file, "%c", vocab[b * max_w + a]);
		  a++;
	  }
    }
	
	fprintf(vocab_file, ",%lld\n", cursor);

	// String end.
    vocab[b * max_w + a] = 0;
	fseek(f, size * sizeof(float), SEEK_CUR);
	cursor += size * sizeof(float);
  }
  fclose(f);
  fclose(vocab_file);
  return 0;
}
