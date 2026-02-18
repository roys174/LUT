CC = gcc
CFLAGS = -O2 -Wall -Wno-unused-result
LDFLAGS = -lm

main: main.c
	$(CC) $(CFLAGS) -o main main.c $(LDFLAGS)

debug: main.c
	$(CC) -g -O0 -Wall -fsanitize=address -o main main.c $(LDFLAGS)

clean:
	rm -f main

.PHONY: clean debug
