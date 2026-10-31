// tests.c : testbench for Arena, Matrix and the network code in MAIN.c
//
// build (gcc):
//   gcc -std=c11 -Wall -Wextra -D__USE_MINGW_ANSI_STDIO=1 tests.c Arena.c Matrix.c Layers.c -o tests.exe -lm
// do not add MAIN.c to that command: tests.c includes it (see below)
// do not add tests.c to the Visual Studio project: it has its own main
//
// exit code 0 = all checks passed, 1 = at least one check failed

#include <string.h>

// MAIN.c holds cost, Normalizer and its own main(), so rename that main to reuse the real code here
// the layer code is in Layers.c / Layers.h (MAIN.c includes Layers.h); MAIN.c still holds cost and Normalizer
#define main mlinc_main
#include "MAIN.c"
#undef main


// ---- test framework --------------------------------------------------------

static int checks_run = 0;
static int checks_failed = 0;
static bool test_failed = false;

#define CHECK(cond) do { \
	checks_run++; \
	if (!(cond)) { \
		checks_failed++; test_failed = true; \
		printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
	} \
} while (0)

#define CHECK_NEAR(actual, expected, tol) do { \
	float actual_ = (actual), expected_ = (expected); \
	checks_run++; \
	if (fabsf(actual_ - expected_) > (tol)) { \
		checks_failed++; test_failed = true; \
		printf("    FAIL %s:%d: %s = %f, expected %f\n", __FILE__, __LINE__, #actual, actual_, expected_); \
	} \
} while (0)

// CHECK_MATRIX(m, rows, columns, values in row order...)
#define CHECK_MATRIX(m, rows, columns, ...) \
	check_matrix_at(__FILE__, __LINE__, #m, (m), (rows), (columns), (float[]){ __VA_ARGS__ })

static void check_matrix_at(const char* file, int line, const char* name, Matrix m, int rows, int columns, const float* expected) {
	checks_run++;
	bool ok = m.matrix && m.rows == rows && m.columns == columns;
	for (int i = 0; ok && i < rows; i++) {
		for (int j = 0; ok && j < columns; j++) {
			if (fabsf(m.matrix[GET(&m, i, j)] - expected[i * columns + j]) > 1e-5f) ok = false;
		}
	}
	if (!ok) {
		checks_failed++; test_failed = true;
		printf("    FAIL %s:%d: %s (%dx%d, expected %dx%d) is:\n", file, line, name, m.rows, m.columns, rows, columns);
		if (m.matrix) matrix_print(m);
	}
}

#define RUN(test) do { \
	test_failed = false; \
	test(); \
	printf("  %s  %s\n", test_failed ? "FAIL" : "pass", #test); \
} while (0)


// ---- arena -----------------------------------------------------------------

static void test_arena_alignment(void) {
	Arena* arena = arena_init(0);
	size_t sizes[] = { 1, 3, 12, 16, 17, 24 };
	for (int i = 0; i < 6; i++) {
		void* p = arena_alloc(arena, sizes[i]);
		CHECK(p != NULL);
		CHECK((uintptr_t)p % ARENA_PADDING_SIZE == 0);
	}
	arena_free(&arena);
}

static void test_arena_exact_fit(void) {
	Arena* arena = arena_init(32);
	CHECK(arena_alloc(arena, 16) != NULL);
	CHECK(arena_alloc(arena, 12) != NULL);	// padded to 16, fills the arena
	CHECK(arena->arena_space_left == 0);
	arena_free(&arena);
}

static void test_arena_full_returns_null(void) {
	Arena* arena = arena_init(32);
	CHECK(arena_alloc(arena, 20) != NULL);	// 32 with padding
	size_t left = arena->arena_space_left;
	printf("    (the next line is expected)\n    ");
	CHECK(arena_alloc(arena, 1) == NULL);
	CHECK(arena->arena_space_left == left);
	arena_free(&arena);
}

static void test_arena_reset(void) {
	Arena* arena = arena_init(64);
	void* first = arena_alloc(arena, 40);
	arena_reset(arena);
	CHECK(arena->arena_space_occupied == 0);
	CHECK(arena->arena_space_left == 64);
	CHECK(arena_alloc(arena, 64) == first);
	arena_free(&arena);
}

static void test_arena_free_sets_null(void) {
	Arena* arena = arena_init(0);
	arena_free(&arena);
	CHECK(arena == NULL);
}


// ---- matrix ----------------------------------------------------------------

static Arena* arena;	// shared by the matrix and network tests, reset before each test

static Matrix from_values(int rows, int columns, const float* values) {
	Matrix m = matrix_init(arena, rows, columns, 0);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			matrix_set(&m, i, j, values[i * columns + j]);
		}
	}
	return m;
}
#define M(rows, columns, ...) from_values((rows), (columns), (float[]){ __VA_ARGS__ })

static float twice(float x) { return 2 * x; }

static void test_matrix_init(void) {
	Matrix m = matrix_init(arena, 2, 3, 1.5f);
	CHECK(m.n == 6);
	CHECK(!m.transpose);
	CHECK_MATRIX(m, 2, 3, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f);
	CHECK_MATRIX(matrix_init_ident(arena, 2, 3), 2, 2, 3, 0, 0, 3);
}

static void test_matrix_mult(void) {
	Matrix a = M(2, 2, 1, 2, 3, 4);
	Matrix b = M(2, 2, 5, 6, 7, 8);
	CHECK_MATRIX(matrix_mult(arena, a, b, NULL), 2, 2, 19, 22, 43, 50);

	Matrix c = M(2, 3, 1, 2, 3, 4, 5, 6);
	Matrix v = M(3, 1, 1, 0, -1);
	CHECK_MATRIX(matrix_mult(arena, c, v, NULL), 2, 1, -2, -2);

	Matrix store = matrix_init(arena, 4, 4, 0);	// bigger than needed
	matrix_mult(NULL, a, b, &store);
	CHECK_MATRIX(store, 2, 2, 19, 22, 43, 50);
}

static void test_matrix_mult_transposed(void) {
	Matrix a = M(2, 3, 1, 2, 3, 4, 5, 6);
	Matrix at = matrix_get_transpose(a);	// 3x2: [1 4; 2 5; 3 6]
	Matrix ones = M(2, 1, 1, 1);
	CHECK_MATRIX(matrix_mult(arena, at, ones, NULL), 3, 1, 5, 7, 9);
	CHECK_MATRIX(matrix_mult(arena, a, at, NULL), 2, 2, 14, 32, 32, 77);
}

static void test_matrix_mult_aliasing(void) {
	Matrix a = M(2, 2, 1, 2, 3, 4);
	Matrix two = matrix_init_ident(arena, 2, 2);
	matrix_mult(arena, a, two, &a);
	CHECK_MATRIX(a, 2, 2, 2, 4, 6, 8);

	Matrix b = M(2, 2, 1, 2, 3, 4);
	matrix_mult(arena, b, b, &b);
	CHECK_MATRIX(b, 2, 2, 7, 10, 15, 22);
}

static void test_matrix_elementwise(void) {
	Matrix a = M(2, 2, 1, 2, 3, 4);
	Matrix b = M(2, 2, 5, 6, 7, 8);
	CHECK_MATRIX(matrix_add(arena, a, b, NULL), 2, 2, 6, 8, 10, 12);
	CHECK_MATRIX(matrix_hadamard_mult(arena, a, b, NULL), 2, 2, 5, 12, 21, 32);
	CHECK_MATRIX(matrix_scale(arena, a, NULL, -0.5f), 2, 2, -0.5f, -1, -1.5f, -2);
	CHECK_MATRIX(matrix_func_apply(arena, a, NULL, twice), 2, 2, 2, 4, 6, 8);

	matrix_add(NULL, a, b, &a);	// in place, as backprop does for weights
	CHECK_MATRIX(a, 2, 2, 6, 8, 10, 12);
}

static void test_matrix_elementwise_transposed(void) {
	Matrix a = M(2, 3, 1, 2, 3, 4, 5, 6);
	Matrix at = matrix_get_transpose(a);
	Matrix b = M(3, 2, 10, 20, 30, 40, 50, 60);
	CHECK_MATRIX(matrix_add(arena, at, b, NULL), 3, 2, 11, 24, 32, 45, 53, 66);
	CHECK_MATRIX(matrix_hadamard_mult(arena, at, b, NULL), 3, 2, 10, 80, 60, 200, 150, 360);
	CHECK_MATRIX(matrix_scale(arena, at, NULL, 2), 3, 2, 2, 8, 4, 10, 6, 12);
	CHECK_MATRIX(matrix_func_apply(arena, at, NULL, twice), 3, 2, 2, 8, 4, 10, 6, 12);

	Matrix store = matrix_init(arena, 3, 2, 0);
	matrix_func_apply(NULL, at, &store, twice);
	CHECK_MATRIX(store, 3, 2, 2, 8, 4, 10, 6, 12);
}

static void test_matrix_transpose(void) {
	Matrix a = M(2, 3, 1, 2, 3, 4, 5, 6);
	CHECK_MATRIX(matrix_get_transpose(a), 3, 2, 1, 4, 2, 5, 3, 6);
	CHECK_MATRIX(matrix_get_transpose(matrix_get_transpose(a)), 2, 3, 1, 2, 3, 4, 5, 6);

	matrix_transpose(&a);
	CHECK_MATRIX(a, 3, 2, 1, 4, 2, 5, 3, 6);

	Matrix copy = matrix_init(arena, 3, 2, 0);
	matrix_copy_into(a, &copy);	// copy of a transposed view is row-major
	CHECK(!copy.transpose);
	CHECK_MATRIX(copy, 3, 2, 1, 4, 2, 5, 3, 6);
}

static void test_matrix_set(void) {
	Matrix m = matrix_init(arena, 2, 3, 0);
	matrix_set(&m, 1, 2, 9);
	matrix_row_set(&m, 0, 1);
	matrix_column_set(&m, 1, 5);
	CHECK_MATRIX(m, 2, 3, 1, 5, 1, 0, 5, 9);

	Matrix t = matrix_get_transpose(M(2, 3, 0, 0, 0, 0, 0, 0));
	matrix_set(&t, 2, 0, 7);	// view position (2,0) = original (0,2)
	CHECK(t.matrix[2] == 7);
}

static void test_matrix_vector_mag(void) {
	CHECK_NEAR(matrix_vector_mag(M(2, 1, 3, 4)), 5, 1e-6f);
	CHECK_NEAR(matrix_vector_mag(M(3, 1, 0, 0, 0)), 0, 1e-6f);
}


// ---- network ---------------------------------------------------------------

// same as layers_init but silent: hidden layers use leaky_relu, the last layer is linear
static Layers make_net(const LayerSize* shape, int count) {
	Layers net;
	net.layers = arena_alloc(arena, sizeof(Layer) * count);
	net.layer_number = count;
	for (int i = 0; i < count; i++) {
		bool last = i == count - 1;
		layer_init(arena, net.layers + i, shape[i].inputs, shape[i].outputs, last ? linear : leaky_relu, last ? linear_prime : leaky_relu_prime);
	}
	return net;
}

// the loss that layer_backprop minimizes: 0.5 * |y - target|^2
static float half_squared_error(Layers net, Matrix x, Matrix target) {
	Matrix y = layers_forward(net, x);
	float sum = 0;
	for (int i = 0; i < y.rows; i++) {
		float d = y.matrix[GET(&y, i, 0)] - target.matrix[GET(&target, i, 0)];
		sum += d * d;
	}
	return 0.5f * sum;
}

static void test_forward_known_values(void) {
	Layer layer;
	layer_init(arena, &layer, 2, 1, linear, linear_prime);
	matrix_set(&layer.weights, 0, 0, 1);
	matrix_set(&layer.weights, 0, 1, 2);
	matrix_set(&layer.biases, 0, 0, 0.5f);
	Layers net = { &layer, 1 };
	Matrix x = M(2, 1, 1, 1);

	CHECK_MATRIX(layers_forward(net, x), 1, 1, 3.5f);	// 1 + 2 + 0.5

	layer.activation = leaky_relu;
	layer.activation_prime = leaky_relu_prime;
	matrix_set(&layer.weights, 0, 0, -1);
	matrix_set(&layer.weights, 0, 1, -1);
	CHECK_MATRIX(layers_forward(net, x), 1, 1, -1.5f * LEAKY_RELU_SLOPE);	// z = -1 - 1 + 0.5 = -1.5
	CHECK_MATRIX(layer.output_buffer_activation_prime, 1, 1, LEAKY_RELU_SLOPE);
}

static void test_backprop_one_step(void) {
	// y = 1 + 2 = 3, target 0, error = 3, grad W = error * x^T = [3 3]
	Layer layer;
	layer_init(arena, &layer, 2, 1, linear, linear_prime);
	matrix_set(&layer.weights, 0, 0, 1);
	matrix_set(&layer.weights, 0, 1, 2);
	matrix_set(&layer.biases, 0, 0, 0);
	Layers net = { &layer, 1 };

	layer_backprop(net, M(2, 1, 1, 1), M(1, 1, 0), 0.1f);
	CHECK_MATRIX(layer.weights, 1, 2, 0.7f, 1.7f);
	CHECK_MATRIX(layer.biases, 1, 1, -0.3f);
}

static void test_normalizer_per_feature(void) {
	// feature 0: 10, 20, 30 (mean 20)   feature 1: 0, 0, 3 (mean 1)   feature 2: constant 5
	Matrix samples[3] = { M(3, 1, 10, 0, 5), M(3, 1, 20, 0, 5), M(3, 1, 30, 3, 5) };
	Normalizer norm = normalizer_fit(arena, samples, 3);
	CHECK_MATRIX(norm.mean, 3, 1, 20, 1, 5);
	CHECK_NEAR(norm.std.matrix[0], sqrtf(200.0f / 3), 1e-5f);
	CHECK_NEAR(norm.std.matrix[1], sqrtf(2), 1e-5f);
	CHECK(norm.std.matrix[2] == 1);	// constant feature: no divide by 0

	// after normalizing, each feature has mean 0 and std 1 (the constant one stays 0)
	Matrix out[3];
	for (int i = 0; i < 3; i++) {
		out[i] = matrix_init(arena, 3, 1, 0);
		normalize(norm, samples[i], &out[i]);
	}
	for (int r = 0; r < 2; r++) {
		float mean = 0, variance = 0;
		for (int i = 0; i < 3; i++) mean += out[i].matrix[r] / 3;
		for (int i = 0; i < 3; i++) variance += (out[i].matrix[r] - mean) * (out[i].matrix[r] - mean) / 3;
		CHECK_NEAR(mean, 0, 1e-5f);
		CHECK_NEAR(variance, 1, 1e-5f);
	}
	CHECK(out[0].matrix[2] == 0 && out[2].matrix[2] == 0);

	// two different samples stay different (per-sample normalization would make them equal)
	CHECK(out[0].matrix[0] != out[2].matrix[0]);
}

static void test_normalizer_round_trip(void) {
	Matrix samples[2] = { M(2, 1, 10, 100), M(2, 1, 30, 400) };
	Normalizer norm = normalizer_fit(arena, samples, 2);
	Matrix normalized = matrix_init(arena, 2, 1, 0);
	Matrix back = matrix_init(arena, 2, 1, 0);
	normalize(norm, samples[1], &normalized);
	CHECK_MATRIX(normalized, 2, 1, 1, 1);
	denormalize(norm, normalized, &back);
	CHECK_MATRIX(back, 2, 1, 30, 400);
}

typedef struct {
	int checked;
	int skipped;	// parameters where +-eps moves a relu input across 0
	int failed;
	float max_error;
} GradResult;

#define GRAD_EPS 1e-2f
#define GRAD_TOL 1e-3f

static int param_count(Layers net) {
	int n = 0;
	for (int l = 0; l < net.layer_number; l++) n += net.layers[l].weights.n + net.layers[l].biases.n;
	return n;
}

static float* param_at(Layers net, int k) {
	for (int l = 0; l < net.layer_number; l++) {
		Layer* layer = net.layers + l;
		if (k < layer->weights.n) return layer->weights.matrix + k;
		k -= layer->weights.n;
		if (k < layer->biases.n) return layer->biases.matrix + k;
		k -= layer->biases.n;
	}
	return NULL;
}

// sign of every pre-activation z after the last forward pass
static void z_signs(Layers net, char* out) {
	for (int l = 0; l < net.layer_number; l++) {
		Matrix z = net.layers[l].output_buffer;
		for (int i = 0; i < z.n; i++) *out++ = z.matrix[i] > 0;
	}
}

// compares the gradient from layer_backprop with (L(p + eps) - L(p - eps)) / 2eps for every parameter p
static GradResult gradient_check(Layers net, Matrix x, Matrix target) {
	GradResult result = { 0 };
	int n = param_count(net);
	int z_n = 0;
	for (int l = 0; l < net.layer_number; l++) z_n += net.layers[l].output_buffer.n;

	float* saved = malloc(n * sizeof(float));
	float* analytic = malloc(n * sizeof(float));
	char* signs_base = malloc(z_n);
	char* signs_plus = malloc(z_n);
	char* signs_minus = malloc(z_n);

	// with learning rate 1, backprop changes each parameter by exactly -gradient
	for (int k = 0; k < n; k++) saved[k] = *param_at(net, k);
	layer_backprop(net, x, target, 1.0f);
	for (int k = 0; k < n; k++) {
		analytic[k] = saved[k] - *param_at(net, k);
		*param_at(net, k) = saved[k];
	}

	half_squared_error(net, x, target);
	z_signs(net, signs_base);

	for (int k = 0; k < n; k++) {
		float* p = param_at(net, k);

		*p = saved[k] + GRAD_EPS;
		float loss_plus = half_squared_error(net, x, target);
		z_signs(net, signs_plus);

		*p = saved[k] - GRAD_EPS;
		float loss_minus = half_squared_error(net, x, target);
		z_signs(net, signs_minus);

		*p = saved[k];

		// at a relu kink the loss has no single slope, so the check means nothing there
		if (memcmp(signs_base, signs_plus, z_n) || memcmp(signs_base, signs_minus, z_n)) {
			result.skipped++;
			continue;
		}

		float numeric = (loss_plus - loss_minus) / (2 * GRAD_EPS);
		float scale = fmaxf(1.0f, fmaxf(fabsf(numeric), fabsf(analytic[k])));
		float error = fabsf(numeric - analytic[k]) / scale;

		result.checked++;
		if (error > result.max_error) result.max_error = error;
		if (error > GRAD_TOL) result.failed++;
	}

	free(saved);
	free(analytic);
	free(signs_base);
	free(signs_plus);
	free(signs_minus);
	return result;
}

static const LayerSize grad_shape[] = { {3, 4}, {4, 4}, {4, 2} };	// 3 layers, so the middle loop of layer_backprop runs

static void test_gradient_check(void) {
	srand(1);
	Layers net = make_net(grad_shape, 3);
	Matrix x = M(3, 1, 0.5f, -0.3f, 0.8f);
	Matrix target = M(2, 1, 0.2f, -0.4f);

	GradResult r = gradient_check(net, x, target);
	printf("    checked %d parameters, skipped %d at a relu kink, max error %.2e (limit %.0e)\n",
		r.checked, r.skipped, r.max_error, GRAD_TOL);
	CHECK(r.checked > 0);
	CHECK(r.failed == 0);
}

// the check must also catch a wrong gradient, or a pass means nothing
static void test_gradient_check_detects_wrong_derivative(void) {
	srand(1);
	Layers net = make_net(grad_shape, 3);
	Matrix x = M(3, 1, 0.5f, -0.3f, 0.8f);
	Matrix target = M(2, 1, 0.2f, -0.4f);

	for (int l = 0; l < net.layer_number - 1; l++) net.layers[l].activation_prime = linear_prime;	// 1 instead of LEAKY_RELU_SLOPE for z < 0

	GradResult r = gradient_check(net, x, target);
	printf("    with a wrong relu derivative: %d of %d parameters fail\n", r.failed, r.checked);
	CHECK(r.failed > 0);
}

static void test_training_reduces_loss(void) {
	srand(2);
	Layers net = make_net(grad_shape, 3);
	Matrix x = M(3, 1, 0.5f, -0.3f, 0.8f);
	Matrix target = M(2, 1, 0.2f, -0.4f);

	float before = half_squared_error(net, x, target);
	for (int i = 0; i < 200; i++) layer_backprop(net, x, target, 0.05f);
	float after = half_squared_error(net, x, target);

	printf("    loss %.4f -> %.6f after 200 steps\n", before, after);
	CHECK(after < 0.01f * before);
}


// ---- main ------------------------------------------------------------------

#define RUN_WITH_ARENA(test) do { arena_reset(arena); RUN(test); } while (0)

int main(void) {
	printf("arena\n");
	RUN(test_arena_alignment);
	RUN(test_arena_exact_fit);
	RUN(test_arena_full_returns_null);
	RUN(test_arena_reset);
	RUN(test_arena_free_sets_null);

	arena = arena_init(1 << 16);

	printf("matrix\n");
	RUN_WITH_ARENA(test_matrix_init);
	RUN_WITH_ARENA(test_matrix_mult);
	RUN_WITH_ARENA(test_matrix_mult_transposed);
	RUN_WITH_ARENA(test_matrix_mult_aliasing);
	RUN_WITH_ARENA(test_matrix_elementwise);
	RUN_WITH_ARENA(test_matrix_elementwise_transposed);
	RUN_WITH_ARENA(test_matrix_transpose);
	RUN_WITH_ARENA(test_matrix_set);
	RUN_WITH_ARENA(test_matrix_vector_mag);

	printf("network\n");
	RUN_WITH_ARENA(test_forward_known_values);
	RUN_WITH_ARENA(test_backprop_one_step);
	RUN_WITH_ARENA(test_normalizer_per_feature);
	RUN_WITH_ARENA(test_normalizer_round_trip);
	RUN_WITH_ARENA(test_gradient_check);
	RUN_WITH_ARENA(test_gradient_check_detects_wrong_derivative);
	RUN_WITH_ARENA(test_training_reduces_loss);

	arena_free(&arena);

	printf("\n%d / %d checks passed\n", checks_run - checks_failed, checks_run);
	return checks_failed ? 1 : 0;
}
