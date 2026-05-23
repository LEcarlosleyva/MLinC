
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include "Arena.h"
#include "Matrix.h"




extern Arena arena;


typedef struct {

	Matrix weights;

	size_t input_size;
	size_t output_size;

	float (*filter)(float);

} Layer;





int main() {
	srand(45);
	arena_init(&arena, 0);
	

	Matrix a;
	matrix_init_rand(&a, 3, 3);


	printf("matrix a = \n");
	matrix_print(a);
	printf("-----------\n");



	Matrix b;
	matrix_init_rand(&b, 3, 3);
	
	printf("matrix b = \n");
	matrix_print(b);
	printf("-----------\n");


	Matrix c = matrix_mult(a, b);
	printf("matrix c = \n");
	matrix_print(c);
	printf("-----------\n");
	c = matrix_scale(c, 3);

	printf("matrix c = \n");
	matrix_print(c);
	printf("-----------\n");

	for (int i = 0; i < 15; i++) {
		c = matrix_mult(c, b);

		c = matrix_add(c, c);
	}

	printf("matrix c = \n");
	matrix_print(c);
	printf("-----------\n");

	printf("\n bytes left in arena = %zu \n, the space occupied is = %zu \n",arena.arena_space_left,arena.arena_space_occupied);

	arena_free(&arena);
}