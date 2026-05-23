



#include <stdio.h>
#include <math.h>
#include <stdlib.h>


float p = 10;
float dt = 0.0005;


//float parameters[] = {12,0.2,0.0,0.4,-0.5,0.32,0.1,1,23,4,0.1,-0.4,-0.5,0.32,0.1,1,23,4,0.1,-0.4 };
float biases[] = { 0.1,-0.4,0.7843,0.413154,-0.4125 };
float a[] = { 0.413,0.4273,0.1234,-0.43251,0.45763584 };


//int parameters_n = sizeof(parameters) / sizeof(parameters[0]);
int biases_n = sizeof(biases) / sizeof(biases[0]);
int a_n = sizeof(a) / sizeof(a[0]);

float inputs[] = {
	1,2,3,4,5,6,7,8,9,10

};


float solutions[] = {
	1,2,3,2,1,0,-1,-3,-4,-3


};
int n = sizeof(inputs) / sizeof(inputs[0]);

inline float relu(float x) {
	return x < 0.0 ? 0.0 : x > 10 ? 10 : x;
}

inline float step(float x) {
	return x < 0.0 ? 0.0 : x < 10 ? 1 : 0;
}

float get_output_prediction(float input) {
	float result = 0;

	for (int i = 0; i < a_n; i++) {
		result += a[i] * relu(input /** parameters[i]*/ + biases[i]);
	}

	
	return result;

}



void normalize(float* v, int n) {
	float mag = 0;

	for (int i = 0; i < n; i++) {
		mag += pow(v[i], 2.0);
	}
	
	mag = sqrt(mag);
	//printf(" the mag is %f \n", mag);
	for (int i = 0; i < n; i++) {
		v[i] /= mag;
		//printf("%f \n", v[i]);
	}


}


//
//
//void get_descent_direction(float * direction, int direction_n) {
//	for(int x = 0; x < n;x++){
//		for (int i = 0; i < direction_n; i++) {
//			direction[i] += (get_output_prediction(inputs[x]) - solutions[x])
//				*step(parameters[i] *  inputs[x] + biases[i]) * inputs[x]*a[i];
//		}
//	}
//
//	/*for (int i = 0; i < n; i++) {
//		direction[i] /= (float) n;
//
//	}*/
//
//	normalize(direction, direction_n);
//
//}
void get_descent_biases(float* bias, int biases_n) {
	for (int x = 0; x < n; x++) {
		for (int i = 0; i < biases_n; i++) {
			bias[i] += (get_output_prediction(inputs[x]) - solutions[x])
				* step(/*parameters[i] * */inputs[x] + biases[i]) * a[i];


		}
	}

	/*for (int i = 0; i < n; i++) {
		bias[i] /= (float)n;

	}*/

	normalize(bias, biases_n);

}

void get_descent_a(float* as, int a_n) {
	for (int x = 0; x < n; x++) {
		for (int i = 0; i < biases_n; i++) {
			as[i] += (get_output_prediction(inputs[x]) - solutions[x])
				* relu(/*parameters[i] * */inputs[x] + biases[i]);


		}
	}

	/*for (int i = 0; i < n; i++) {
		bias[i] /= (float)n;

	}*/

	normalize(as, a_n);

}

void parameters_update(float * parameters,float * direction,float dt, int parameters_n) {
	for (int i = 0; i < parameters_n; i++) {
		parameters[i] -= direction[i] * dt;

	}

}


int train = 1000000;

int main() {

	//float * directions = malloc(sizeof(float)*parameters_n);
	float * bias = malloc(sizeof(float) * biases_n);
	float * as   = malloc(sizeof(float) * a_n);


 	for (int i = 0; i < a_n; i++) {
		//directions[i] = 0;
		bias[i] = 0;
		as[i] = 0;
	}


	
	for (int i = 0; i < train; i++) {
		//get_descent_direction(directions, parameters_n);
		get_descent_biases(bias, a_n);
		get_descent_a(as, a_n);

		//parameters_update(parameters, directions, 0.0001, parameters_n);
		parameters_update(biases, bias, 0.1, a_n);
		parameters_update(a, as, 0.05, a_n);
	}

	/*printf("the final param are : ");
	for (int i = 0; i < parameters_n; i++){
		printf("%f ,",parameters[i]);
	}*/
	printf("\n");
	printf("the final biases are : ");
	for (int i = 0; i < a_n; i++) {
		printf("%f ,", biases[i]);
	}
	printf("\n");
	printf("the final as are : ");
	for (int i = 0; i < a_n; i++) {
		printf("%f ,", a[i]);
	}
	printf("\n");
	for (int i = 0; i < n;i++) {
		printf("f(%f) = %f \n", inputs[i],get_output_prediction(inputs[i]));
	}




	return 0;
}