

#include "Matrix.h"





Matrix matrix_init(Arena * arena , int rows, int columns, float init_val) {
	float* r = arena_alloc(arena, rows * columns * sizeof(float));

	if (!r) {
		printf("failed to arena_alloc space for matrix in matrix_init\n");
		exit(1);
		return;
	};
	
	Matrix matr;

	matr.matrix = r;
	matr.rows = rows;
	matr.columns = columns;
	matr.n = rows * columns;
	matr.transpose = false;


	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			r[GET(&matr, i, j)] = init_val;
		}

	}

	return matr;


}

Matrix matrix_init_rand(Arena * arena, int rows, int columns) {
	float* r = arena_alloc(arena, rows * columns * sizeof(float));
	if (!r) {
		printf("failed to arena_alloc space for matrix in matrix_init_rand\n");
		exit(1);
		return;
	}
	Matrix matr;
	matr.matrix = r;
	matr.rows = rows;
	matr.columns = columns;
	matr.n = rows * columns;
	matr.transpose = false;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			r[GET(&matr, i, j)] = (float)(rand() % 100 - 50 ) / 50;
		}

	}

	return matr;

}

Matrix matrix_init_ident(Arena * arena, int n, float scalar) {
	float* r = arena_alloc(arena, n *n * sizeof(float));
	
	if (!r) {
		printf("failed to arena_alloc space for matrix in matrix_init_ident\n");
		exit(1);
		return;
	}
	Matrix matr;
	matr.matrix = r;
	matr.rows = n;
	matr.columns = n;
	matr.n = n * n;
	matr.transpose = false;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			r[GET(&matr, i, j)] = i == j ? scalar : 0.0;
		}

	}

	return matr;

}


void matrix_print(Matrix matr) {
	
	if (!matr.matrix) {
		printf("the matrix \n");
		return;
	}

	for (int i = 0; i < matr.rows; i++) {
	
		printf("[");
		for (int j = 0; j < matr.columns - 1; j++) {
			printf("%+3.5f ,", matr.matrix[GET(&matr, i, j)]);

		}
		printf("%+3.5f ]\n", matr.matrix[GET(&matr, i, matr.columns - 1)]);

	}
}





Matrix matrix_mult(Arena * arena,Matrix a, Matrix b,Matrix * store){

	Matrix* c = store;
	
	if (a.columns != b.rows) {
		
		printf("the number of columns (inputs ) in a does not match the number of rows (outputs) in b in matrix_mult \n");
		exit(1);
		
		return;

	}

	if (!a.matrix){
		printf("the matrix a is empty in matrix_mult\n");
		exit(1);
	}
	if(!b.matrix){
		printf("the matrix b is empty in matrix_mult\n");
		exit(1);
		return;
	}




	if(!c) { // if storage is null create and return a new matrix 

		c = arena_alloc(arena, sizeof(Matrix));
		*c = matrix_init(arena, a.rows, b.columns, 0);
	} else { // otherwise just use storage to store the resulting matrix as long as it can in fact soore it 

	if(!c->matrix) {
		printf("the matrix store is empty in matrix_mult\n");
		exit(1);
		return;
	}

	if (c->n < a.rows * b.columns) {
		printf("not enough space in c to store the product in matrix_mult\n");
		exit(1);
		return;
	}
	c->rows = a.rows;
	c->columns = b.columns;
	c->transpose = false;

	}



	for (int i = 0; i < c->rows; i++) {
		for (int j = 0; j < c->columns; j++) {
			float tmp = 0;
			for (int k = 0; k < a.columns; k++) {
				tmp += a.matrix[GET(&a, i, k)] * b.matrix[GET(&b, k, j)];

			}
			c->matrix[GET(c, i, j)] = tmp;
		}
	}
	
	return *c;// return a shallow copy of the matrix 


}

Matrix matrix_hadamard_mult(Arena* arena,Matrix a, Matrix b, Matrix *store){
	
	Matrix* c = store;
	
	if (a.columns != b.columns) {
		printf("the number of columns in a does not match the number of colums in b in matrix_hadamard_mult \n");
		exit(1);
		return;
	}


	if (a.rows != b.rows) {
		printf("the number of rows in a does not match the number of rows in b in matrix_hadamard_mult \n");
		exit(1);
		return;
	}
	if (!a.matrix) {
		printf("the matrix a is empty inmatrix_hadamard_mult\n");
		exit(1);
	}
	if (!b.matrix) {
		printf("the matrix b is empty in matrix_hadamard_mult\n");
		exit(1);
		return;
	}
	if (!c) {
		c = arena_alloc(arena, sizeof(Matrix));
		*c = matrix_init(arena, a.rows, a.columns, 0);
	}else {
	if (!c->matrix) {
		printf("the matrix c is empty in matrix_hadamard_mult\n");
		exit(1);
		return;
	}
	if (c->n < a.rows * a.columns) {
		printf("not enough space in c to store the product in matrix_hadamard_mult\n");
		exit(1);
		return;
	}

	c->rows = a.rows;
	c->columns = a.columns;
	c->transpose = false;
	}

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.columns; j++) {
			c->matrix[GET(c, i, j)] = a.matrix[GET(&a,i,j)] * b.matrix[GET(&b,i,j)];
		}
	}
	

	return *c;

}



Matrix matrix_add(Arena * arena, Matrix a, Matrix b, Matrix * store) {

	Matrix* c = store;

	if (a.columns != b.columns) {
		printf("the number of columns in a does not match the number of colums in b in matrix_hadamard_mult \n");
		exit(1);
		return;
	}


	if (a.rows != b.rows) {
		printf("the number of rows in a does not match the number of rows in b in matrix_hadamard_mult \n");
		exit(1);
		return;
	}
	if (!a.matrix) {
		printf("the matrix a is empty inmatrix_hadamard_mult\n");
		exit(1);
	}
	if (!b.matrix) {
		printf("the matrix b is empty in matrix_hadamard_mult\n");
		exit(1);
		return;
	}
	if (!c) {
		c = arena_alloc(arena, sizeof(Matrix));
		*c = matrix_init(arena, a.rows, a.columns, 0);
	} else {
	if (!c->matrix) {
		printf("the matrix c is empty in matrix_hadamard_mult\n");
		exit(1);
		return;
	}
	if (c->n < a.rows * a.columns) {
		printf("not enough space in c to store the product in matrix_hadamard_mult\n");
		exit(1);
		return;
	}

	}
	c->rows = a.rows;
	c->columns = a.columns;
	c->transpose = false;

	for (int i = 0; i < c->rows; i++) {
		for (int j = 0; j < c->columns; j++) {

			c->matrix[GET(c, i, j)] = a.matrix[GET(&a, i, j)] + b.matrix[GET(&b, i, j)];

		}

	}
	return *c;
}




Matrix matrix_scale(Arena * arena,Matrix a, Matrix  * store, float scalar) {
	Matrix* b = store;
	if (!a.matrix) {
		printf("matrix a is empty in matrix_scale \n");
		exit(1);
		return;
	}
		
	if (!b) {
		b = arena_alloc(arena, sizeof(Matrix));
		*b = matrix_init(arena, a.rows, a.columns, 0);

	} else {

	if (!b->matrix) {
		printf("the matrix b is empty in matrix_scale\n");
		exit(1);
		return;
	}
	if (b->n < a.n) {
		printf("not enough space in b to store the product in matrix_scale\n");
		exit(1);
	}

	b->rows = a.rows;
	b->columns = a.columns;
	b->transpose = false;
	
	}
	for (int i = 0; i < b->rows; i++) {
		for (int j = 0; j < b->columns; j++) {
			b->matrix[GET(b, i, j)] = a.matrix[GET(&a, i, j)] * scalar;
		}
	}

	return *b;

}


void matrix_transpose(Matrix* c) {
	if (!c) {
		printf("the pointer to the sotarage matrix is empty in matrix_transpose\n");
		exit(1);
		return;
	}
	if (!c->matrix) {
		printf("the matrix c is empty in matrix_hadamard_mult\n");
		exit(1);
		return;
	}
	int tmp = c->columns;
	c->columns = c->rows;
	c->rows = tmp;
	c->transpose = !c->transpose;
}

Matrix matrix_get_transpose(Matrix a) {
	if (!a.matrix) {
		printf("the matrix c is empty in matrix_hadamard_mult\n");
		exit(1);
		return;
	}

	Matrix retval = a;
	retval.rows = a.columns;
	retval.columns = a.rows;
	retval.transpose = !a.transpose;

	return retval;
}

void matrix_copy_into(Matrix a, Matrix * b) {

	if (!a.matrix) {
		printf("matrix a is empty in matrix_scale \n");
		exit(1);
		return;
	}

	if (!b) {
		printf("the pointer to the sotarage matrix is empty in matrix_scale\n");
		exit(1);
		return;

	}

	if (!b->matrix) {
		printf("the matrix b is empty in matrix_scale\n");
		exit(1);
		return;
	}
	if (b->n < a.n) {
		printf("not enough space in b to store the product in matrix_scale\n");
		exit(1);
	}
	b->columns = a.columns;
	b->rows = a.rows;
	b->transpose = false;

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.columns; j++) {
			b->matrix[GET(b, i, j)] = a.matrix[GET(&a, i, j)];
		}
	}


}



Matrix matrix_func_apply(Arena * arena, Matrix a,Matrix * store, float (*f)(float)) {
	
	Matrix* b = store;

	if (!a.matrix) {
		printf("matrix a is empty in matrix_func_apply \n");
		exit(1);
		return;
	}

	if (!b) {
		
		b = arena_alloc(arena, sizeof(Matrix));
		*b = matrix_init(arena, a.rows, a.columns, 0);
	} else {

	if (!b->matrix) {
		printf("the matrix b is empty in matrix_func_apply\n");
		exit(1);
		return;
	}
	if (b->n < a.n) {
		printf("not enough space in b to store the product in matrix_func_apply\n");
		exit(1);
	}

	b->rows = a.rows;
	b->columns = a.columns;
	b->transpose = false;
	
	}
	if (!f) {
		printf("f is not valid in matrix_func_apply\n");
		exit(1);
		return;
	}
	for (int i = 0; i < a.rows * a.columns; i++) {

		b->matrix[i] = f(a.matrix[i]);
	}

	return *b;

}



void matrix_set(Matrix * a,int i, int j , float val) {
	if (!a) {
		printf("the pointer to the sotarage matrix is empty in matrix_transpose\n");
		exit(1);
		return;
	}
	if (!a->matrix) {
		printf("the matrix c is empty in matrix_hadamard_mult\n");
		exit(1);
		return;
	}

	if (i >= a->rows) {
		printf("index i out of range in matrix_set\n");
		exit(1);
		return;
	}

	if (j >= a->columns) {
		printf("index j out of range in matrix_set\n");
		exit(1);
		return;
	}

	a->matrix[GET(a, i, j)] = val;

}


void matrix_row_set(Matrix* a, int i, float val) {
	if (!a) {
		printf("the pointer to the sotarage matrix is empty in matrix_transpose\n");
		exit(1);
		return;
	}
	if (!a->matrix) {
		printf("the matrix c is empty in matrix_hadamard_mult\n");
		exit(1);
		return;
	}

	if (i >= a->rows) {
		printf("index i out of range in matrix_set\n");
		exit(1);
		return;
	}

	for (int j = 0; j < a->columns; j++){

		a->matrix[GET(a, i, j)] = val;
	}

	
}


void matrix_column_set(Matrix* a, int j, float val) {


	if (!a) {
		printf("the pointer to the sotarage matrix is empty in matrix_transpose\n");
		exit(1);
		return;
	}
	if (!a->matrix) {
		printf("the matrix c is empty in matrix_hadamard_mult\n");
		exit(1);
		return;
	}

	if (j >= a->columns) {
		printf("index j out of range in matrix_set\n");
		exit(1);
		return;
	}

	for (int i = 0; i < a->rows; i++) {

		a->matrix[GET(a, i, j)] = val;
	}







}


float matrix_vector_mag(Matrix a) {

	if (!a.matrix) {
		printf("the matrix a is empty in matrix_vector_mag\n");
		exit(1);
		return;
	}

	if (a.columns != 1) {
		printf("the matrix a is not a vector in matrix_vector_mag\n");
		exit(1);
		return;
	}


	float result = 0;

	for (int i = 0; i < a.rows;i++) {
		
		result += a.matrix[GET(&a, i, 0)] * a.matrix[GET(&a, i, 0)];
		
	}


	return sqrt(result);


}