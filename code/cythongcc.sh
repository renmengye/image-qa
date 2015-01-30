gcc -shared -pthread -fPIC -fwrapv -I/usr/include/python2.7 -O3 -mtune=native -o $2.so $1.c
