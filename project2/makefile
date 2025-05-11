MPICC = mpicc

CFLAGS = -std=c99 -Wall
LDFLAGS = -llapack -lblas

all: rps

rps: rps.c
	$(MPICC) $(CFLAGS) -o rps rps.c $(LDFLAGS)

clean:
	rm -f rps