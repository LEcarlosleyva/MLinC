#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include "Arena.h"

#define MATRIX_BUFFER_DEFAULT_ROW_SIZE 10
#define MATRIX_BUFFER_DEFAULT_COLUMN_SIZE 10

#define GET(M,X,Y) (((M)->transpose) ? ((X) + (Y)*(M)->rows) : ((X)*(M)->columns + (Y))) 

typedef struct {
	float* matrix;
	int rows;
	int columns;
	int n;
	bool transpose;
} Matrix;



Matrix matrix_init(Arena* arena, int rows, int columns, float init_val);
Matrix matrix_init_rand(Arena* arena, int rows, int columns);
Matrix matrix_init_ident(Arena* arena, int n, float scalar);
void matrix_print(Matrix matr);
Matrix matrix_mult(Arena* arena, Matrix a, Matrix b, Matrix* store);
Matrix matrix_hadamard_mult(Arena* arena, Matrix a, Matrix b, Matrix* store);
Matrix matrix_add(Arena* arena, Matrix a, Matrix b, Matrix* store);
Matrix matrix_scale(Arena* arena, Matrix a, Matrix* store, float scalar);
void matrix_transpose(Matrix* c);
Matrix matrix_get_transpose(Matrix a);
void matrix_copy_into(Matrix a, Matrix* b);
Matrix matrix_func_apply(Arena* arena, Matrix a, Matrix* store, float (*f)(float));
void matrix_set(Matrix* a, int i, int j, float val);
void matrix_row_set(Matrix* a, int i, float val);
void matrix_column_set(Matrix* a, int j, float val);
float matrix_vector_mag(Matrix a);

