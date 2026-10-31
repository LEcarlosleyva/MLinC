#pragma once

#include <stddef.h>
#include "Arena.h"
#include "Matrix.h"

typedef struct {

	Matrix weights;
	Matrix output_buffer;
	Matrix output_buffer_activation;
	Matrix output_buffer_activation_prime;
	
	Matrix biases;

	size_t input_size;
	size_t output_size;

	float (*activation)(float);
	float (*activation_prime)(float);

} Layer;

typedef struct {
	Layer* layers;
	int layer_number;
} Layers;



typedef struct {
	int inputs;
	int outputs;
} LayerSize;

#define LEAKY_RELU_SLOPE 0.3	// double on purpose: the old code computed 0.3*x in double, and 0.3f changes the training result

// activation functions and their derivatives
float leaky_relu(float x);
float leaky_relu_prime(float x);
float linear(float x);
float linear_prime(float x);

// layer functions
void layer_init(Arena* arena, Layer* l, int input_size, int output_size, float (*activation)(float), float (*activation_prime)(float));
Layers layers_init(Arena* arena, LayerSize* layer_sizes, int layer_number);
Matrix layers_forward(Layers layers, Matrix input);
void layers_print(Layers layers);
void layer_backprop(Layers layers, Matrix input, Matrix target, float learning_rate);
