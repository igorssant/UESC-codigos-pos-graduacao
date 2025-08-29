CC=gcc
CFLAGS=-static -fPIC -Wall -Wextra
LDFLAGS=-shared
TARGET_LIB=src/core/libgradient.so

all: $(TARGET_LIB)

$(TARGET_LIB): gradient.o
	$(CC) $(LDFLAGS) -o $@ gradient.o

gradient.o: src/lib/gradient.c src/lib/gradient.h
	$(CC) $(CFLAGS) -c $< -o $@ -lm

clean:
	rm -f *.o $(TARGET_LIB)

to_deploy:
	rm -f *.o
