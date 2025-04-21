test: test.o
	gcc -o test test.o -lgomp

test.o: test.c
	gcc -c test.c -fopenmp

clean:
	rm -f test test.o
