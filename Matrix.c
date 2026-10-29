#include <stdarg.h>
#include "Matrix.h"


// ---- helpers ---------------------------------------------------------------
// every check prints the name of the calling function (func) and stops the program

static void matrix_fail(const char* func, const char* fmt, ...) {
	va_list args;
	va_start(args, fmt);
	fprintf(stderr, "error: ");
	vfprintf(stderr, fmt, args);
	va_end(args);
	fprintf(stderr, " in %s\n", func);
	exit(1);
}
#define MATRIX_FAIL(...) matrix_fail(__func__, __VA_ARGS__)

static void check_not_null(const void* p, const char* name, const char* func) {
	if (!p) matrix_fail(func, "pointer %s is null", name);
}
#define CHECK_NOT_NULL(p) check_not_null((p), #p, __func__)

static void check_not_empty(Matrix m, const char* name, const char* func) {
	if (!m.matrix) matrix_fail(func, "matrix %s is empty", name);
}
#define CHECK_NOT_EMPTY(m) check_not_empty((m), #m, __func__)

static void check_same_shape(Matrix a, Matrix b, const char* func) {
	if (a.rows != b.rows || a.columns != b.columns)
		matrix_fail(func, "shape %dx%d does not match %dx%d", a.rows, a.columns, b.rows, b.columns);
}

static void check_index(int index, int limit, const char* name, const char* func) {
	if (index < 0 || index >= limit) matrix_fail(func, "index %s = %d out of range [0, %d)", name, index, limit);
}
#define CHECK_INDEX(i, limit) check_index((i), (limit), #i, __func__)

// allocates the float buffer, values are not set
static Matrix matrix_alloc(Arena* arena, int rows, int columns, const char* func) {
	if (rows <= 0 || columns <= 0) matrix_fail(func, "invalid size %dx%d", rows, columns);

	float* r = arena_alloc(arena, (size_t)rows * columns * sizeof(float));
	if (!r) matrix_fail(func, "failed to arena_alloc a %dx%d matrix", rows, columns);

	Matrix matr;
	matr.matrix = r;
	matr.rows = rows;
	matr.columns = columns;
	matr.n = rows * columns;
	matr.transpose = false;
	return matr;
}

// store == NULL : makes a new rows x columns matrix in the arena
// store != NULL : checks store is big enough and sets its shape to rows x columns (row-major)
static Matrix ensure_store(Arena* arena, Matrix* store, int rows, int columns, const char* func) {
	if (!store) return matrix_alloc(arena, rows, columns, func);

	check_not_empty(*store, "store", func);
	if (store->n < rows * columns)
		matrix_fail(func, "store too small (%d < %d)", store->n, rows * columns);

	store->rows = rows;
	store->columns = columns;
	store->transpose = false;
	return *store;
}

static Matrix matrix_copy(Arena* arena, Matrix a, const char* func) {
	Matrix copy = matrix_alloc(arena, a.rows, a.columns, func);
	matrix_copy_into(a, &copy);
	return copy;
}


// ---- init ------------------------------------------------------------------

Matrix matrix_init(Arena* arena, int rows, int columns, float init_val) {
	Matrix matr = matrix_alloc(arena, rows, columns, __func__);

	for (int i = 0; i < matr.n; i++) {
		matr.matrix[i] = init_val;
	}
	return matr;
}

Matrix matrix_init_rand(Arena* arena, int rows, int columns) {
	Matrix matr = matrix_alloc(arena, rows, columns, __func__);

	for (int i = 0; i < matr.n; i++) {
		matr.matrix[i] = (float)(rand() % 100 - 50) / 50;
	}
	return matr;
}

Matrix matrix_init_ident(Arena* arena, int n, float scalar) {
	Matrix matr = matrix_alloc(arena, n, n, __func__);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matr.matrix[GET(&matr, i, j)] = i == j ? scalar : 0.0f;
		}
	}
	return matr;
}


void matrix_print(Matrix matr) {

	if (!matr.matrix) {
		printf("the matrix is empty\n");
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


// ---- operations ------------------------------------------------------------
// pattern: check inputs, check shape, get the output with ensure_store, do the work

Matrix matrix_mult(Arena* arena, Matrix a, Matrix b, Matrix* store) {
	CHECK_NOT_EMPTY(a);
	CHECK_NOT_EMPTY(b);
	if (a.columns != b.rows)
		MATRIX_FAIL("columns of a (%d) do not match rows of b (%d)", a.columns, b.rows);

	Matrix c = ensure_store(arena, store, a.rows, b.columns, __func__);

	// the loop writes c while it still reads a and b, so an input that shares memory with c is copied first
	if (c.matrix == a.matrix || c.matrix == b.matrix) {
		if (!arena) MATRIX_FAIL("store shares memory with an input: pass an arena for the copy");
		if (c.matrix == a.matrix) a = matrix_copy(arena, a, __func__);
		if (c.matrix == b.matrix) b = matrix_copy(arena, b, __func__);
	}

	for (int i = 0; i < c.rows; i++) {
		for (int j = 0; j < c.columns; j++) {
			float tmp = 0;
			for (int k = 0; k < a.columns; k++) {
				tmp += a.matrix[GET(&a, i, k)] * b.matrix[GET(&b, k, j)];
			}
			c.matrix[GET(&c, i, j)] = tmp;
		}
	}
	return c;
}

Matrix matrix_hadamard_mult(Arena* arena, Matrix a, Matrix b, Matrix* store) {
	CHECK_NOT_EMPTY(a);
	CHECK_NOT_EMPTY(b);
	check_same_shape(a, b, __func__);

	Matrix c = ensure_store(arena, store, a.rows, a.columns, __func__);

	for (int i = 0; i < c.rows; i++) {
		for (int j = 0; j < c.columns; j++) {
			c.matrix[GET(&c, i, j)] = a.matrix[GET(&a, i, j)] * b.matrix[GET(&b, i, j)];
		}
	}
	return c;
}

Matrix matrix_add(Arena* arena, Matrix a, Matrix b, Matrix* store) {
	CHECK_NOT_EMPTY(a);
	CHECK_NOT_EMPTY(b);
	check_same_shape(a, b, __func__);

	Matrix c = ensure_store(arena, store, a.rows, a.columns, __func__);

	for (int i = 0; i < c.rows; i++) {
		for (int j = 0; j < c.columns; j++) {
			c.matrix[GET(&c, i, j)] = a.matrix[GET(&a, i, j)] + b.matrix[GET(&b, i, j)];
		}
	}
	return c;
}

Matrix matrix_scale(Arena* arena, Matrix a, Matrix* store, float scalar) {
	CHECK_NOT_EMPTY(a);

	Matrix b = ensure_store(arena, store, a.rows, a.columns, __func__);

	for (int i = 0; i < b.rows; i++) {
		for (int j = 0; j < b.columns; j++) {
			b.matrix[GET(&b, i, j)] = a.matrix[GET(&a, i, j)] * scalar;
		}
	}
	return b;
}

Matrix matrix_func_apply(Arena* arena, Matrix a, Matrix* store, float (*f)(float)) {
	CHECK_NOT_EMPTY(a);
	if (!f) MATRIX_FAIL("function f is null");

	Matrix b = ensure_store(arena, store, a.rows, a.columns, __func__);

	for (int i = 0; i < b.rows; i++) {
		for (int j = 0; j < b.columns; j++) {
			b.matrix[GET(&b, i, j)] = f(a.matrix[GET(&a, i, j)]);
		}
	}
	return b;
}

void matrix_copy_into(Matrix a, Matrix* store) {
	CHECK_NOT_EMPTY(a);
	CHECK_NOT_NULL(store);

	Matrix b = ensure_store(NULL, store, a.rows, a.columns, __func__);

	for (int i = 0; i < b.rows; i++) {
		for (int j = 0; j < b.columns; j++) {
			b.matrix[GET(&b, i, j)] = a.matrix[GET(&a, i, j)];
		}
	}
}


// ---- transpose -------------------------------------------------------------

void matrix_transpose(Matrix* c) {
	CHECK_NOT_NULL(c);
	CHECK_NOT_EMPTY(*c);

	int tmp = c->columns;
	c->columns = c->rows;
	c->rows = tmp;
	c->transpose = !c->transpose;
}

Matrix matrix_get_transpose(Matrix a) {
	CHECK_NOT_EMPTY(a);

	Matrix retval = a;
	retval.rows = a.columns;
	retval.columns = a.rows;
	retval.transpose = !a.transpose;
	return retval;
}


// ---- set -------------------------------------------------------------------

void matrix_set(Matrix* a, int i, int j, float val) {
	CHECK_NOT_NULL(a);
	CHECK_NOT_EMPTY(*a);
	CHECK_INDEX(i, a->rows);
	CHECK_INDEX(j, a->columns);

	a->matrix[GET(a, i, j)] = val;
}

void matrix_row_set(Matrix* a, int i, float val) {
	CHECK_NOT_NULL(a);
	CHECK_NOT_EMPTY(*a);
	CHECK_INDEX(i, a->rows);

	for (int j = 0; j < a->columns; j++) {
		a->matrix[GET(a, i, j)] = val;
	}
}

void matrix_column_set(Matrix* a, int j, float val) {
	CHECK_NOT_NULL(a);
	CHECK_NOT_EMPTY(*a);
	CHECK_INDEX(j, a->columns);

	for (int i = 0; i < a->rows; i++) {
		a->matrix[GET(a, i, j)] = val;
	}
}


// ---- vector ----------------------------------------------------------------

float matrix_vector_mag(Matrix a) {
	CHECK_NOT_EMPTY(a);
	if (a.columns != 1) MATRIX_FAIL("matrix a is not a column vector (%dx%d)", a.rows, a.columns);

	float result = 0;
	for (int i = 0; i < a.rows; i++) {
		result += a.matrix[GET(&a, i, 0)] * a.matrix[GET(&a, i, 0)];
	}
	return sqrt(result);
}
