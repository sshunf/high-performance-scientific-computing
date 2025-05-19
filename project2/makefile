MPICC = mpicc

CFLAGS = -std=c99 -Wall -O2 -march=native -ffast-math -funroll-loops
LDFLAGS = -llapack -lblas

all: rps

rps: rps.c
	$(MPICC) $(CFLAGS) -o rps rps.c $(LDFLAGS)

clean:
	rm -f rps