#include <stdlib.h>
#include <stdio.h>
#include "Layers.h"

float leaky_relu(float x) {
	return x > 0 ? x : LEAKY_RELU_SLOPE * x;
}

float leaky_relu_prime(float x) {
	return x > 0 ? 1 : LEAKY_RELU_SLOPE;

}

float linear(float x) {
	return x;
}

float linear_prime(float x) {
	return 1;
}

void layer_init(Arena * arena,Layer * l,int input_size,int output_size, float (*activation)(float),float (*activation_prime)(float)) {
	l->input_size = input_size;
	l->output_size = output_size;
	
	l->weights = matrix_init_rand(arena, l->output_size, l->input_size);
	l->biases = matrix_init_rand(arena, l->output_size, 1);

	l->output_buffer = matrix_init(arena, l->output_size, 1, 0);
	l->output_buffer_activation = matrix_init(arena,  l->output_size, 1, 0);
	l->output_buffer_activation_prime = matrix_init(arena,  l->output_size, 1, 0);
	
	
	l->activation = activation;
	l->activation_prime = activation_prime;

	

}






Layers layers_init(Arena* arena, LayerSize* layer_sizes, int layer_number) {
	Layers layers;

	layers.layers = arena_alloc(arena, sizeof(Layer) * layer_number);
	layers.layer_number = layer_number;

	for (int i = 0; i < layer_number - 1; i++) {
		layer_init(arena, layers.layers + i, layer_sizes[i].inputs, layer_sizes[i].outputs, leaky_relu, leaky_relu_prime);
		printf("layer number %d has %d inputs and %d outputs the buffer has %d outputs the layer weigts are \n", i + 1, layers.layers[i].weights.columns, layers.layers[i].weights.rows, layers.layers[i].output_buffer.rows);
		matrix_print(layers.layers[i].weights);
		printf("\n");

	}
	int i = layer_number - 1;
	layer_init(arena, layers.layers + i, layer_sizes[i].inputs, layer_sizes[i].outputs, linear, linear_prime);
	printf("layer number %d has %d inputs and %d outputs the buffer has %d outputs the layer weigts are \n", i + 1, layers.layers[i].weights.columns, layers.layers[i].weights.rows, layers.layers[i].output_buffer.rows);
	matrix_print(layers.layers[i].weights);
	printf("\n");


	return layers;
}
Matrix layers_forward(Layers layers, Matrix input) {

	
	Matrix tmp;


	tmp = input;
	

	for (int j = 0; j < layers.layer_number; j++) {

		matrix_mult(NULL, layers.layers[j].weights, tmp, &layers.layers[j].output_buffer);
		matrix_add(NULL, layers.layers[j].output_buffer, layers.layers[j].biases, &layers.layers[j].output_buffer);

		matrix_func_apply(NULL, layers.layers[j].output_buffer, &layers.layers[j].output_buffer_activation, layers.layers[j].activation);
		matrix_func_apply(NULL, layers.layers[j].output_buffer, &layers.layers[j].output_buffer_activation_prime, layers.layers[j].activation_prime);
		tmp = layers.layers[j].output_buffer_activation;
	}
	
	return tmp;


}

void layers_print(Layers layers) {

	for (int i = 0; i < layers.layer_number; i++) {
		printf("the weights of layer number %d are \n", i);
		matrix_print(layers.layers[i].weights);
		printf("the biases of layer number %d are \n", i);
		matrix_print(layers.layers[i].biases);

	}



}

void layer_backprop(Layers layers, Matrix input, Matrix target, float learning_rate) {

	//printf("layers_backprop-----------------------------------------------------------------------\n");
	Arena* scratch_space = arena_init(0);


	Matrix neg_target = matrix_scale(scratch_space, target, NULL, -1);

	Matrix output = layers_forward(layers, input);



	Matrix error = matrix_add(scratch_space, output, neg_target, NULL);
	
	Matrix grad;
	

	error = matrix_hadamard_mult(scratch_space, error, layers.layers[layers.layer_number - 1].output_buffer_activation_prime, NULL);
	
	for (int i = layers.layer_number - 1; i != 0; i--) {
	
		Matrix output_activation_buffer_t = matrix_get_transpose(layers.layers[i - 1].output_buffer_activation);
		

		grad = matrix_mult(scratch_space, error, output_activation_buffer_t, NULL);
		
		Matrix biases_correction = matrix_scale(scratch_space, error, NULL, -learning_rate);

		matrix_add(NULL, layers.layers[i].biases, biases_correction, &layers.layers[i].biases);

		Matrix w_t = matrix_get_transpose(layers.layers[i].weights);
	
		error = matrix_mult(scratch_space, w_t, error,NULL);
	
		Matrix output_activation_prime_buffer = layers.layers[i - 1].output_buffer_activation_prime;
	
		error = matrix_hadamard_mult(scratch_space, error, output_activation_prime_buffer, NULL);
	
		Matrix w_correction = matrix_scale(scratch_space, grad, NULL, -learning_rate);
		


		matrix_add(NULL, layers.layers[i].weights, w_correction, &layers.layers[i].weights);
		
	}
	/*printf("the current layer is is %d ----------------------\n ", 0);*/
	
	Matrix biases_correction = matrix_scale(scratch_space, error, NULL, -learning_rate);

	matrix_add(NULL, layers.layers[0].biases, biases_correction, &layers.layers[0].biases);
	
	Matrix output_activation_buffer_t = matrix_get_transpose(input);
	
	grad = matrix_mult(scratch_space, error, output_activation_buffer_t, NULL);

	

	Matrix w_correction = matrix_scale(scratch_space, grad, NULL, -learning_rate);


	matrix_add(NULL, layers.layers[0].weights, w_correction, &layers.layers[0].weights);
	
	arena_free(&scratch_space);
}
