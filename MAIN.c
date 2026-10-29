
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
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
	int K;
	int V;
} Pair;

float relu(float x) {
	return x > 0 ? x : 0.3*x;
}

float relu_prime(float x) {
	return x > 0 ? 1 : 0.3;

}

float linear(float x) {
	return x;
}

float linear_prime(float x) {
	return 1;
}

float cost(Matrix y, Matrix y_prime) {
	Arena* scratch_space = arena_init(0);
	
	Matrix neg_y_prime = matrix_scale(scratch_space, y_prime, NULL, -1);
	
	Matrix error = matrix_add(scratch_space, y, neg_y_prime, NULL);
	
	float result = matrix_vector_mag(error);
	
	arena_free(&scratch_space);

	return result;

}

void layer_init(Arena * arena,Layer * l,int output_size,int input_size, float (*activation)(float),float (*activation_prime)(float)) {
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






Layers layers_init(Arena* arena, Pair* internal_structure, int layer_number) {
	Layers layers;

	layers.layers = arena_alloc(arena, sizeof(Layer) * layer_number);
	layers.layer_number = layer_number;

	for (int i = 0; i < layer_number - 1; i++) {
		layer_init(arena, layers.layers + i, internal_structure[i].V, internal_structure[i].K, relu, relu_prime);
		printf("layer number %d has %d inputs and %d outputs the buffer has %d outputs the layer weigts are \n", i + 1, layers.layers[i].weights.columns, layers.layers[i].weights.rows, layers.layers[i].output_buffer.rows);
		matrix_print(layers.layers[i].weights);
		printf("\n");

	}
	int i = layer_number - 1;
	layer_init(arena, layers.layers + i, internal_structure[i].V, internal_structure[i].K, linear, linear_prime);
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

void layer_backprop(Layers layers, Matrix input, Matrix y_prime, float learning_rate) {

	//printf("layers_backprop-----------------------------------------------------------------------\n");
	Arena* scratch_space = arena_init(0);
	

	Matrix neg_y = matrix_scale(scratch_space, y_prime, NULL, -1);
	
	Matrix y = layers_forward(layers, input);
	

	
	Matrix error = matrix_add(scratch_space, y, neg_y, NULL);
	
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
;

	matrix_add(NULL, layers.layers[0].weights, w_correction, &layers.layers[0].weights);
	
	arena_free(&scratch_space);
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
	Pair internal_structure[] = { {2,10},{10,10} ,{10,3} };
	int layer_number = sizeof(internal_structure) / sizeof(internal_structure[0]);
	float cost_av = 0;



	Layers layers = layers_init(arena, internal_structure, layer_number);
	



	printf("input initializatio-----------------------------------------------------------------------------\n");
	// input declaration and initialization 
	Matrix inputs[20];

	int input_number = sizeof(inputs) / sizeof(inputs[0]);

	for (int i = 0; i < input_number; i++) {
		inputs[i] = matrix_init(arena_for_input, internal_structure[0].K, 1, i + 10 );
		matrix_row_set(inputs + i, 1, i + 11);
		printf("\n");
		matrix_print(inputs[i]);
	}

	// input declaration and initialization

	
	printf("output initializatio-----------------------------------------------------------------------------\n");

	Matrix * y = arena_alloc(arena_for_input,sizeof(Matrix)*input_number) ;

	for (int i = 0; i < input_number; i++) {
		y[i] = matrix_init(arena_for_input, internal_structure[layer_number - 1].V, 1, (i + 1) *i );
		matrix_row_set(y + i, 1, i * 3);
		printf("\n");
		matrix_print(y[i]);


	}

	Matrix tmp;
	printf("layers forward test-----------------------------------------------------------------------------\n");


	for (int i = 0; i < input_number; i++) {
		tmp = layers_forward(layers, inputs[i]);
		printf("the input is \n");
		matrix_print(inputs[i]);
		printf("the output is \n");
		matrix_print(tmp);
		cost_av += cost(y[i], tmp);
	}
	cost_av /= input_number;
	printf("the average cost is %f \n", cost_av);



	layers_print(layers);
	printf("training-----------------------------\n----------\n-------------------------------------\n");

	int training_iteration = 99900;
	float learning_rate = 0.000003;
	for (int i = 0; i < training_iteration; i++) {
		
		for (int i = 0; i < input_number; i++) {
			layer_backprop(layers, inputs[i], y[i], learning_rate);
		}



	}

	layers_print(layers);



	printf("layers forward test after training-----------------------------------------------------------------------------\n");

	cost_av = 0;
	
	for (int i = 0; i < input_number; i++) {
		tmp = layers_forward(layers, inputs[i]);
		printf("the input is \n");
		matrix_print(inputs[i]);
		printf("the output is \n");
		matrix_print(tmp);
		cost_av += cost(y[i], tmp);
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




