
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "Arena.h"
#include "Matrix.h"
#include "Layers.h"






float cost(Matrix target, Matrix output) {
	Arena* scratch_space = arena_init(0);

	Matrix neg_output = matrix_scale(scratch_space, output, NULL, -1);

	Matrix error = matrix_add(scratch_space, target, neg_output, NULL);
	
	float result = matrix_vector_mag(error);
	
	arena_free(&scratch_space);

	return result;

}

// per-feature normalization: each row (feature) of the samples gets mean 0 and std 1 over all samples
typedef struct {
	Matrix mean;
	Matrix std;
} Normalizer;

Normalizer normalizer_fit(Arena* arena, Matrix* samples, int sample_number) {
	int features = samples[0].rows;
	Normalizer norm;
	norm.mean = matrix_init(arena, features, 1, 0);
	norm.std = matrix_init(arena, features, 1, 0);

	for (int r = 0; r < features; r++) {
		float mean = 0;
		for (int i = 0; i < sample_number; i++) {
			mean += samples[i].matrix[GET(&samples[i], r, 0)];
		}
		mean /= sample_number;

		float variance = 0;
		for (int i = 0; i < sample_number; i++) {
			float d = samples[i].matrix[GET(&samples[i], r, 0)] - mean;
			variance += d * d;
		}
		variance /= sample_number;

		matrix_set(&norm.mean, r, 0, mean);
		matrix_set(&norm.std, r, 0, variance > 0 ? sqrtf(variance) : 1);	// a constant feature has no information, and this avoids dividing by 0
	}
	return norm;
}

// store = (x - mean) / std
void normalize(Normalizer norm, Matrix x, Matrix* store) {
	if (x.rows != norm.mean.rows || store->rows != x.rows) {
		printf("size mismatch in normalize\n");
		exit(1);
	}
	for (int r = 0; r < x.rows; r++) {
		matrix_set(store, r, 0, (x.matrix[GET(&x, r, 0)] - norm.mean.matrix[r]) / norm.std.matrix[r]);
	}
}

// store = x * std + mean, the inverse of normalize
void denormalize(Normalizer norm, Matrix x, Matrix* store) {
	if (x.rows != norm.mean.rows || store->rows != x.rows) {
		printf("size mismatch in denormalize\n");
		exit(1);
	}
	for (int r = 0; r < x.rows; r++) {
		matrix_set(store, r, 0, x.matrix[GET(&x, r, 0)] * norm.std.matrix[r] + norm.mean.matrix[r]);
	}
}





int main() {

	srand(0);
	Arena *arena = arena_init(0);
	Arena* arena_for_input = arena_init(0);
	arena_report(*arena);
	arena_report(*arena_for_input);




	printf("layer initialization----------------------------------------------------------------------------\n");


	//		----One column of the parameter matrix is used as a bias and the final row is used to calculate the output buffer for the layer which would be 1;
	//   xxxx CORRECTION NOTE : biases are not necessary when the newtork is massive , since other weights can simulate biases and the inputs are normalized 
	// for small networks biases are essential otherwise we wont be able to learn functions that pass throug zero
	// TODO investigate how to use cuda to do matrix multiplication
	LayerSize layer_sizes[] = { {2,10},{10,10} ,{10,3} };	// {inputs, outputs} per layer
	int layer_number = sizeof(layer_sizes) / sizeof(layer_sizes[0]);
	float cost_av = 0;



	Layers layers = layers_init(arena, layer_sizes, layer_number);
	



	printf("input initializatio-----------------------------------------------------------------------------\n");
	// input declaration and initialization 
	Matrix inputs[20];

	int input_number = sizeof(inputs) / sizeof(inputs[0]);

	for (int i = 0; i < input_number; i++) {
		inputs[i] = matrix_init(arena_for_input, layer_sizes[0].inputs, 1, i + 10 );
		matrix_row_set(inputs + i, 1, i + 11);
		printf("\n");
		matrix_print(inputs[i]);
	}

	// input declaration and initialization

	
	printf("output initializatio-----------------------------------------------------------------------------\n");

	Matrix * targets = arena_alloc(arena_for_input,sizeof(Matrix)*input_number) ;

	for (int i = 0; i < input_number; i++) {
		targets[i] = matrix_init(arena_for_input, layer_sizes[layer_number - 1].outputs, 1, (i + 1) *i );
		matrix_row_set(targets + i, 1, i * 3);
		printf("\n");
		matrix_print(targets[i]);


	}

	printf("normalization-----------------------------------------------------------------------------\n");
	// the network trains on normalized data; outputs and cost are printed in original units

	Normalizer input_norm = normalizer_fit(arena_for_input, inputs, input_number);
	Normalizer target_norm = normalizer_fit(arena_for_input, targets, input_number);
	printf("input mean and std per feature\n");
	matrix_print(input_norm.mean);
	matrix_print(input_norm.std);
	printf("target mean and std per feature\n");
	matrix_print(target_norm.mean);
	matrix_print(target_norm.std);

	Matrix* inputs_norm = arena_alloc(arena_for_input, sizeof(Matrix) * input_number);
	Matrix* targets_norm = arena_alloc(arena_for_input, sizeof(Matrix) * input_number);

	for (int i = 0; i < input_number; i++) {
		inputs_norm[i] = matrix_init(arena_for_input, inputs[i].rows, 1, 0);
		normalize(input_norm, inputs[i], &inputs_norm[i]);
		targets_norm[i] = matrix_init(arena_for_input, targets[i].rows, 1, 0);
		normalize(target_norm, targets[i], &targets_norm[i]);
	}

	Matrix prediction = matrix_init(arena_for_input, layer_sizes[layer_number - 1].outputs, 1, 0);	// network output in original units

	Matrix tmp;
	printf("layers forward test-----------------------------------------------------------------------------\n");


	for (int i = 0; i < input_number; i++) {
		tmp = layers_forward(layers, inputs_norm[i]);
		denormalize(target_norm, tmp, &prediction);
		printf("the input is \n");
		matrix_print(inputs[i]);
		printf("the output is \n");
		matrix_print(prediction);
		cost_av += cost(targets[i], prediction);
	}
	cost_av /= input_number;
	printf("the average cost is %f \n", cost_av);



	layers_print(layers);
	printf("training-----------------------------\n----------\n-------------------------------------\n");

	int training_iteration = 2000;	// was 99900 at learning rate 0.000003 on raw data
	float learning_rate = 0.01f;
	for (int epoch = 0; epoch < training_iteration; epoch++) {

		for (int i = 0; i < input_number; i++) {
			layer_backprop(layers, inputs_norm[i], targets_norm[i], learning_rate);
		}



	}

	layers_print(layers);



	printf("layers forward test after training-----------------------------------------------------------------------------\n");

	cost_av = 0;
	
	for (int i = 0; i < input_number; i++) {
		tmp = layers_forward(layers, inputs_norm[i]);
		denormalize(target_norm, tmp, &prediction);
		printf("the input is \n");
		matrix_print(inputs[i]);
		printf("the output is \n");
		matrix_print(prediction);
		cost_av += cost(targets[i], prediction);
	}
	cost_av /= input_number;
	printf("the average cost is %f \n", cost_av);

 // Memeroy managment
	arena_report(*arena);
	arena_free(&arena);
	arena_report(*arena_for_input);
	arena_free(&arena_for_input);
	

	return 0;
}




