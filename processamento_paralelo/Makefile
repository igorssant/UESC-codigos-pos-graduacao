CC = gcc
CFLAGS = -Wall -O3# -fopenmp
TARGET = bin/sequential
SRC = src/sequential_implementation.c src/config.c src/tensor.c
LIB = lib
args =

all: compile run

compile:
	$(CC) -o $(TARGET) $(CFLAGS) $(SRC) -I$(LIB)

run:
	./$(TARGET) $(args)

clean:
	rm -f $(TARGET)

.PHONY: all compile run clean

