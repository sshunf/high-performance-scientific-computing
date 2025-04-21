#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

int main()
{
	double a = pow(17.0, 0.25);
	double b = 2*sin(M_PI/4)*cos(M_PI/4);
       
	double* array = (double*)malloc(sizeof(double)*5);
	array[0] = M_PI;
	array[1] = M_E;
	array[3] = a;
	array[4] = b;	
	
	double* third = &array[2];
	*third = M_SQRT2;
	int size = 5;
	
	for (int i = 0; i < size; ++i) {
		printf("index %d, val: %f\n", i, array[i]);
	}

	free(array);

	return 0;
}
