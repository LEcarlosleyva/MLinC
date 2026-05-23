#pragma once

#include <stdbool.h>
#include "Arena.h"

#define MATRIX_BUFFER_DEFAULT_ROW_SIZE 10
#define MATRIX_BUFFER_DEFAULT_COLUMN_SIZE 10

#define GET(M,X,Y) ((M)->transpose) ? ((X) + (Y)*(M)->rows) : ((X)*(M)->columns + (Y)) 

typedef struct {
	float* matrix;
	int rows;
	int columns;
	bool transpose;
} Matrix;



void matrix_init(Matrix* matr, int rows, int columns, float init_val);
void matrix_init_rand(Matrix* matr, int rows, int columns);
void matrix_print(Matrix matr);
Matrix matrix_mult(Matrix a, Matrix b);
Matrix matrix_add(Matrix a, Matrix b);
Matrix matrix_scale(Matrix a, float scalar);
Matrix matrix_transpose(Matrix c);
Matrix matrix_copy(Matrix a);

