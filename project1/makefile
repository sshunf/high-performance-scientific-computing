stokes: stokes.o
	gcc -o stokes stokes.o -lgomp

stokes.o: stokes.c
	gcc -c stokes.c -fopenmp

clean:
	rm -f stokes stokes.o
