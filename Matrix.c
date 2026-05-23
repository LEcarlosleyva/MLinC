

#include "Matrix.h"



Arena arena;


Matrix matrix_mult_buffer;
Matrix matrix_add_buffer;
Matrix matrix_scale_buffer;
Matrix matrix_transpose_buffer;


void matrix_init(Matrix* matr, int rows, int columns, float init_val) {
	float* r = arena_alloc(&arena, rows * columns * sizeof(float));

	if (!r) return;

	matr->matrix = r;
	matr->rows = rows;
	matr->columns = columns;
	matr->transpose = false;


	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			r[GET(matr, i, j)] = init_val;
		}

	}


}

void matrix_init_rand(Matrix* matr, int rows, int columns) {
	float* r = arena_alloc(&arena, rows * columns * sizeof(float));
	if (!r) return;


	matr->matrix = r;
	matr->rows = rows;
	matr->columns = columns;
	matr->transpose = false;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			r[GET(matr, i, j)] = (float)(rand() % 100 - 50) / 50;
		}

	}


}


void matrix_print(Matrix matr) {
	for (int i = 0; i < matr.rows; i++) {
		printf("[");
		for (int j = 0; j < matr.columns - 1; j++) {
			printf("%+3.2f ,", matr.matrix[GET(&matr, i, j)]);

		}
		printf("%+3.2f ]\n", matr.matrix[GET(&matr, i, matr.columns - 1)]);

	}
}

Matrix matrix_mult(Matrix a, Matrix b) {

	if (!matrix_mult_buffer.matrix) {
		matrix_init(&matrix_mult_buffer, MATRIX_BUFFER_DEFAULT_ROW_SIZE, MATRIX_BUFFER_DEFAULT_COLUMN_SIZE, 0);
	}

	if (a.columns != b.rows) return matrix_mult_buffer;

	if (a.columns * b.rows > matrix_mult_buffer.columns * matrix_mult_buffer.rows) return matrix_mult_buffer; // Manage this case by allocating a bigger buffer

	matrix_mult_buffer.rows = a.rows;
	matrix_mult_buffer.columns = b.columns;
	matrix_mult_buffer.transpose = false;


	for (int i = 0; i < matrix_mult_buffer.rows; i++) {
		for (int j = 0; j < matrix_mult_buffer.columns; j++) {

			for (int k = 0; k < a.columns; k++) {
				matrix_mult_buffer.matrix[GET(&matrix_mult_buffer, i, j)] = a.matrix[GET(&a, i, k)] * b.matrix[GET(&b, k, j)];

			}
		}
	}

	return matrix_mult_buffer;


}


Matrix matrix_add(Matrix a, Matrix b) { // ads a + b to c

	if (!matrix_add_buffer.matrix) {
		matrix_init(&matrix_add_buffer, MATRIX_BUFFER_DEFAULT_ROW_SIZE, MATRIX_BUFFER_DEFAULT_COLUMN_SIZE, 0);
	}


	if (a.rows != b.rows) return matrix_add_buffer;
	if (a.columns != b.columns) return matrix_add_buffer;

	if (a.columns * a.rows > matrix_add_buffer.columns * matrix_add_buffer.rows) return matrix_add_buffer; // Manage this case by allocating a bigger buffer

	matrix_add_buffer.rows = a.rows;
	matrix_add_buffer.columns = a.columns;
	matrix_add_buffer.transpose = false;


	for (int i = 0; i < matrix_add_buffer.rows; i++) {
		for (int j = 0; j < matrix_add_buffer.columns; j++) {
			matrix_add_buffer.matrix[GET(&matrix_add_buffer, i, j)] = a.matrix[GET(&a, i, j)] + b.matrix[GET(&b, i, j)];
		}
	}

	return matrix_add_buffer;

}




Matrix matrix_scale(Matrix a, float scalar) { // use this to create copies 


	if (!matrix_scale_buffer.matrix) {
		matrix_init(&matrix_scale_buffer, MATRIX_BUFFER_DEFAULT_ROW_SIZE, MATRIX_BUFFER_DEFAULT_COLUMN_SIZE, 0);
	}

	if (!a.matrix) return matrix_scale_buffer;

	if (a.columns * a.rows > matrix_scale_buffer.columns * matrix_scale_buffer.rows) return matrix_scale_buffer; // Manage this case by allocating a bigger buffer


	matrix_scale_buffer.rows = a.rows;
	matrix_scale_buffer.columns = a.columns;
	matrix_scale_buffer.transpose = a.transpose;

	for (int i = 0; i < a.rows * a.columns; i++) {
		matrix_scale_buffer.matrix[i] = a.matrix[i] * scalar;
	}

	matrix_scale_buffer.transpose = a.transpose;

	return matrix_scale_buffer;

}

Matrix matrix_transpose(Matrix c) {

	if (!matrix_transpose_buffer.matrix) {
		matrix_init(&matrix_transpose_buffer, MATRIX_BUFFER_DEFAULT_ROW_SIZE, MATRIX_BUFFER_DEFAULT_COLUMN_SIZE, 0);
	}


	if (!c.matrix) return matrix_transpose_buffer;

	if (c.columns * c.rows > matrix_transpose_buffer.columns * matrix_transpose_buffer.rows) return matrix_transpose_buffer; // Manage this case by allocating a bigger buffer

	for (int i = 0; i < c.rows * c.columns; i++) {
		matrix_transpose_buffer.matrix[i] = c.matrix[i];
	}


	matrix_transpose_buffer.rows = c.columns;
	matrix_transpose_buffer.columns = c.rows;
	matrix_transpose_buffer.transpose = true;

	return matrix_transpose_buffer;
}

Matrix matrix_copy(Matrix a) {

	if (!a.matrix) return a;

	Matrix result;

	float* r = arena_alloc(&arena, a.rows * a.columns * sizeof(float));

	result.matrix = r;
	result.rows = a.rows;
	result.columns = a.columns;
	result.transpose = a.transpose;


	if (!r) return result;





	for (int i = 0; i < result.rows * result.columns; i++) {
		result.matrix[i] = a.matrix[i];
	}


	return result;
}