// live loss curves for the MLinC network, drawn with raylib
// it trains the same network on the same data as MAIN.c; with split "none" the final cost must match MAIN.c (2.536326)
//
// build (MinGW gcc, from the MLinC folder):
//   gcc -std=c11 -O2 -D__USE_MINGW_ANSI_STDIO=1 -I../raylib/src viewer/loss_view.c Arena.c Matrix.c
//       -L../raylib/build-mingw -lraylib -lopengl32 -lgdi32 -lwinmm -o loss_view.exe
//
// keys: SPACE pause/run, R restart, T next split mode, UP/DOWN learning rate x2 / /2, S save loss_<split>.png
// run with --auto [none|random|last] to train to the end, save loss_<split>.png, print the final costs and exit

#include <string.h>

// raylib also has a type named Matrix: rename it here, the library is not changed
#define Matrix RaylibMatrix
#include "raylib.h"
#undef Matrix

#define main mlinc_main
#include "../MAIN.c"	// TODO: include Layers.h when the network moves out of MAIN.c
#undef main

#define SAMPLE_NUMBER    20
#define TEST_NUMBER      4	// 20 % of the samples
#define SHUFFLE_SEED     1u
#define MAX_EPOCHS       2000	// same as training_iteration in MAIN.c
#define EPOCHS_PER_FRAME 10
#define MAX_POINTS       (MAX_EPOCHS / EPOCHS_PER_FRAME + 1)

#define WIDTH  960
#define HEIGHT 540
#define PLOT_LEFT   80
#define PLOT_RIGHT  (WIDTH - 30)
#define PLOT_TOP    60
#define PLOT_BOTTOM (HEIGHT - 60)


// ---- data ------------------------------------------------------------------

typedef enum {
	SPLIT_NONE,	// all samples train, no test set: same run as MAIN.c
	SPLIT_RANDOM,	// shuffled, then the last TEST_NUMBER are the test set: tests interpolation
	SPLIT_LAST,	// not shuffled, the last TEST_NUMBER samples are the test set: tests extrapolation
	SPLIT_MODE_NUMBER
} SplitMode;

static const char* split_names[SPLIT_MODE_NUMBER] = { "none", "random", "last" };

typedef struct {
	Matrix inputs[SAMPLE_NUMBER];
	Matrix targets[SAMPLE_NUMBER];
	Matrix inputs_norm[SAMPLE_NUMBER];
	Matrix targets_norm[SAMPLE_NUMBER];
	int count;
} SampleSet;

typedef struct {
	SampleSet train;
	SampleSet test;
	Normalizer input_norm;
	Normalizer target_norm;
	Matrix prediction;
} Dataset;

// own generator: rand() must stay untouched, because srand(0) + rand() gives the start weights
static unsigned shuffle_next(unsigned* state) {
	*state ^= *state << 13;
	*state ^= *state >> 17;
	*state ^= *state << 5;
	return *state;
}

static void set_normalize(SampleSet* set, Normalizer input_norm, Normalizer target_norm, Arena* arena) {
	for (int i = 0; i < set->count; i++) {
		set->inputs_norm[i] = matrix_init(arena, set->inputs[i].rows, 1, 0);
		normalize(input_norm, set->inputs[i], &set->inputs_norm[i]);
		set->targets_norm[i] = matrix_init(arena, set->targets[i].rows, 1, 0);
		normalize(target_norm, set->targets[i], &set->targets_norm[i]);
	}
}

// same samples as main() in MAIN.c, then split into a train set and a test set
static Dataset dataset_make(Arena* arena, int input_size, int output_size, SplitMode mode) {
	Matrix inputs[SAMPLE_NUMBER];
	Matrix targets[SAMPLE_NUMBER];
	for (int i = 0; i < SAMPLE_NUMBER; i++) {
		inputs[i] = matrix_init(arena, input_size, 1, i + 10);
		matrix_row_set(&inputs[i], 1, i + 11);
		targets[i] = matrix_init(arena, output_size, 1, (i + 1) * i);
		matrix_row_set(&targets[i], 1, i * 3);
	}

	// Fisher-Yates on the sample order, so each input stays with its target
	int order[SAMPLE_NUMBER];
	for (int i = 0; i < SAMPLE_NUMBER; i++) order[i] = i;
	if (mode == SPLIT_RANDOM) {
		unsigned state = SHUFFLE_SEED;
		for (int i = SAMPLE_NUMBER - 1; i > 0; i--) {
			int j = shuffle_next(&state) % (i + 1);
			int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
		}
	}

	Dataset d;
	int test_number = mode == SPLIT_NONE ? 0 : TEST_NUMBER;
	d.train.count = SAMPLE_NUMBER - test_number;
	d.test.count = test_number;
	for (int i = 0; i < d.train.count; i++) {
		d.train.inputs[i] = inputs[order[i]];
		d.train.targets[i] = targets[order[i]];
	}
	for (int i = 0; i < d.test.count; i++) {
		d.test.inputs[i] = inputs[order[d.train.count + i]];
		d.test.targets[i] = targets[order[d.train.count + i]];
	}

	// fit on the train set only: the test set must not change the mean and std (data leakage)
	d.input_norm = normalizer_fit(arena, d.train.inputs, d.train.count);
	d.target_norm = normalizer_fit(arena, d.train.targets, d.train.count);
	set_normalize(&d.train, d.input_norm, d.target_norm, arena);
	set_normalize(&d.test, d.input_norm, d.target_norm, arena);

	d.prediction = matrix_init(arena, output_size, 1, 0);
	return d;
}

// average cost in original units, same as the final print of MAIN.c; forward only, no backprop
static float average_cost(Layers layers, SampleSet* set, Dataset* d) {
	float sum = 0;
	for (int i = 0; i < set->count; i++) {
		Matrix out = layers_forward(layers, set->inputs_norm[i]);
		denormalize(d->target_norm, out, &d->prediction);
		sum += cost(set->targets[i], d->prediction);
	}
	return sum / set->count;
}

// srand(0) before layers_init gives the same start weights as MAIN.c
static Layers network_restart(Arena* arena, LayerSize* layer_sizes, int layer_number) {
	arena_reset(arena);
	srand(0);
	return layers_init(arena, layer_sizes, layer_number);
}


// ---- curves ----------------------------------------------------------------

typedef struct {
	int epochs[MAX_POINTS];
	float train[MAX_POINTS];
	float test[MAX_POINTS];
	int point_number;
	bool has_test;
} Curves;

static void curves_add(Curves* c, int epoch, Layers layers, Dataset* d) {
	c->epochs[c->point_number] = epoch;
	c->train[c->point_number] = average_cost(layers, &d->train, d);
	c->test[c->point_number] = c->has_test ? average_cost(layers, &d->test, d) : 0;
	c->point_number++;
}

static void curves_restart(Curves* c, Layers layers, Dataset* d) {
	c->point_number = 0;
	c->has_test = d->test.count > 0;
	curves_add(c, 0, layers, d);
}


// ---- drawing ---------------------------------------------------------------

static float plot_x(int epoch) {
	return PLOT_LEFT + (float)epoch / MAX_EPOCHS * (PLOT_RIGHT - PLOT_LEFT);
}

// log scale: the cost falls about 100x, a linear scale shows the end as a flat line
static float plot_y(float cost_value, float log_min, float log_max) {
	float t = (log10f(cost_value) - log_min) / (log_max - log_min);
	return PLOT_BOTTOM - t * (PLOT_BOTTOM - PLOT_TOP);
}

static void draw_line(const Curves* c, const float* values, float log_min, float log_max, Color color) {
	for (int i = 1; i < c->point_number; i++) {
		Vector2 a = { plot_x(c->epochs[i - 1]), plot_y(values[i - 1], log_min, log_max) };
		Vector2 b = { plot_x(c->epochs[i]), plot_y(values[i], log_min, log_max) };
		DrawLineEx(a, b, 2, color);
	}
}

static void draw_plot(const Curves* c, SplitMode mode, float learning_rate, bool running) {
	// the y range covers whole decades of both lines
	float lo = c->train[0], hi = c->train[0];
	for (int i = 0; i < c->point_number; i++) {
		lo = fminf(lo, c->train[i]);
		hi = fmaxf(hi, c->train[i]);
		if (c->has_test) {
			lo = fminf(lo, c->test[i]);
			hi = fmaxf(hi, c->test[i]);
		}
	}
	float log_min = floorf(log10f(lo));
	float log_max = ceilf(log10f(hi));
	if (log_max <= log_min) log_max = log_min + 1;

	for (int p = (int)log_min; p <= (int)log_max; p++) {
		float y = plot_y(powf(10, (float)p), log_min, log_max);
		DrawLine(PLOT_LEFT, (int)y, PLOT_RIGHT, (int)y, LIGHTGRAY);
		DrawText(TextFormat("1e%d", p), 20, (int)y - 8, 16, DARKGRAY);
	}
	for (int e = 0; e <= MAX_EPOCHS; e += 500) {
		float x = plot_x(e);
		DrawLine((int)x, PLOT_TOP, (int)x, PLOT_BOTTOM, LIGHTGRAY);
		DrawText(TextFormat("%d", e), (int)x - 12, PLOT_BOTTOM + 8, 16, DARKGRAY);
	}
	DrawRectangleLines(PLOT_LEFT, PLOT_TOP, PLOT_RIGHT - PLOT_LEFT, PLOT_BOTTOM - PLOT_TOP, GRAY);
	DrawText("epoch", PLOT_RIGHT - 50, PLOT_BOTTOM + 30, 16, DARKGRAY);

	draw_line(c, c->train, log_min, log_max, BLUE);
	if (c->has_test) draw_line(c, c->test, log_min, log_max, RED);

	int last = c->point_number - 1;
	const char* status = c->epochs[last] >= MAX_EPOCHS ? "done" : running ? "running" : "paused";
	DrawText(TextFormat("split %s   epoch %d   lr %g   %s", split_names[mode], c->epochs[last], learning_rate, status),
		PLOT_LEFT, 12, 20, BLACK);
	DrawText(TextFormat("train %.4f", c->train[last]), PLOT_LEFT, 36, 18, BLUE);
	if (c->has_test) DrawText(TextFormat("test %.4f", c->test[last]), PLOT_LEFT + 180, 36, 18, RED);
	DrawText("SPACE pause  R restart  T split  UP/DOWN lr  S save png", PLOT_LEFT, HEIGHT - 22, 16, GRAY);
}

// draws into an off-screen texture of fixed size, so the png does not depend on the screen or on Windows scaling
static void save_png(RenderTexture2D target, const Curves* c, SplitMode mode, float learning_rate, bool running) {
	BeginTextureMode(target);
	ClearBackground(RAYWHITE);
	draw_plot(c, mode, learning_rate, running);
	EndTextureMode();

	Image image = LoadImageFromTexture(target.texture);
	ImageFlipVertical(&image);	// OpenGL textures are stored bottom row first
	ExportImage(image, TextFormat("loss_%s.png", split_names[mode]));
	UnloadImage(image);
}


// ---- main ------------------------------------------------------------------

int main(int argc, char** argv) {
	bool auto_mode = argc > 1 && strcmp(argv[1], "--auto") == 0;
	SplitMode mode = SPLIT_RANDOM;
	if (auto_mode && argc > 2) {
		for (int m = 0; m < SPLIT_MODE_NUMBER; m++) {
			if (strcmp(argv[2], split_names[m]) == 0) mode = m;
		}
	}

	LayerSize layer_sizes[] = { {2,10},{10,10},{10,3} };	// same as MAIN.c
	int layer_number = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
	int input_size = layer_sizes[0].inputs;
	int output_size = layer_sizes[layer_number - 1].outputs;

	Arena* net_arena = arena_init(0);
	Arena* data_arena = arena_init(0);

	Layers layers = network_restart(net_arena, layer_sizes, layer_number);
	Dataset data = dataset_make(data_arena, input_size, output_size, mode);

	float learning_rate = 0.01f;	// same as MAIN.c
	int epoch = 0;
	bool running = true;
	static Curves curves;	// static: about 5 KB, kept off the stack
	curves_restart(&curves, layers, &data);

	SetTraceLogLevel(LOG_WARNING);
	InitWindow(WIDTH, HEIGHT, "MLinC loss curve");
	SetTargetFPS(60);
	RenderTexture2D png_target = LoadRenderTexture(WIDTH, HEIGHT);

	while (!WindowShouldClose()) {
		if (IsKeyPressed(KEY_SPACE)) running = !running;
		if (IsKeyPressed(KEY_UP)) learning_rate *= 2;
		if (IsKeyPressed(KEY_DOWN)) learning_rate /= 2;
		bool save = IsKeyPressed(KEY_S);
		bool new_split = IsKeyPressed(KEY_T);
		if (new_split) {
			mode = (mode + 1) % SPLIT_MODE_NUMBER;
			arena_reset(data_arena);
			data = dataset_make(data_arena, input_size, output_size, mode);
		}
		if (new_split || IsKeyPressed(KEY_R)) {
			layers = network_restart(net_arena, layer_sizes, layer_number);
			epoch = 0;
			curves_restart(&curves, layers, &data);
		}

		if (running && epoch < MAX_EPOCHS) {
			for (int k = 0; k < EPOCHS_PER_FRAME; k++, epoch++) {
				for (int i = 0; i < data.train.count; i++) {
					layer_backprop(layers, data.train.inputs_norm[i], data.train.targets_norm[i], learning_rate);
				}
			}
			curves_add(&curves, epoch, layers, &data);
		}

		BeginDrawing();
		ClearBackground(RAYWHITE);
		draw_plot(&curves, mode, learning_rate, running);
		EndDrawing();

		if (save || (auto_mode && epoch >= MAX_EPOCHS)) save_png(png_target, &curves, mode, learning_rate, running);

		if (auto_mode && epoch >= MAX_EPOCHS) {
			int last = curves.point_number - 1;
			printf("split %s epoch %d train %f", split_names[mode], epoch, curves.train[last]);
			if (curves.has_test) printf(" test %f", curves.test[last]);
			printf("\n");
			break;
		}
	}

	UnloadRenderTexture(png_target);
	CloseWindow();
	arena_free(&net_arena);
	arena_free(&data_arena);
	return 0;
}
