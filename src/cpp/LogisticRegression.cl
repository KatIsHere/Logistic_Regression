#pragma OPENCL EXTENSION cl_intel_printf : enable
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

// performes an iteration of nesterow gradient calculatoin
__kernel  __attribute__((vec_type_hint(float)))
void nesterow_iteration(__global float* X, 
		__constant float* dX, 
		__global float* Y, 
		const int width,
		const float L, const float l_param) 
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	__local int pos;
	__local float prev;
	pos = i*width + j;
	prev = X[pos];
	X[pos] = mad(-L, dX[pos], Y[pos]);
	Y[pos] = mad(l_param, X[pos] - prev, X[pos]);
}


// performes an iteration of nesterow gradient calculatoin
__kernel  __attribute__((vec_type_hint(float)))
void gradient_descent_iteration(__global float* X, 
		__constant float* dX, 
		const int width,
		const float L) 
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	__local int pos;
	__local float prev;
	pos = i*width + j;
	prev = X[pos];
	X[pos] = mad(-L, dX[pos], prev);
}


// calculates loss of cross enthropy with softmax activation
__kernel __attribute__((vec_type_hint(double)))
void cross_entropy_with_softmax(__constant double* ExpW, 
		__global double* dscores,
		__constant short* y,					// real object class
		__global double* loss,				// overall loss - should be a pointer to an element
		const int num_classes,
		const int num_samples) 			// number of samples
{
	const int row = get_global_id(0);
	__local double sum;
	sum = 0.;
	if(row < num_samples){ 
		for(int i = 0; i < num_classes; ++i){ 
			sum +=  ExpW[row*num_classes + i];
		}
		sum = (sum != 0.0 ? sum : 0.000001);
		for(int i = 0; i < num_classes; ++i){ 
			dscores[row*num_classes + i] = (ExpW[row*num_classes + i] / sum) / num_samples;
		}
		if(y[row] < num_classes){ 
			*loss -= (log(dscores[row*num_classes + y[row]] * num_samples)) / num_samples;	
			dscores[row*num_classes + y[row]] -= 1 / num_samples;	// there should be a spike at the right answer _/\_
		}
	}
}


// los += 0.5 * reg * (W*W)
// adds regularization to loss
__kernel void regularization(__constant float* W,	// found weights
		__global double* loss,							// overall loss
		const float reg,								// regularization param
		const int num_classes)							// number of classes
{
	const int row = get_global_id(0);
	const int col = get_global_id(1);
	__local double sum;
	sum = W[row*num_classes + col] * W[row*num_classes + col];
	*loss += 0.5 * reg * sum;
}


// performes A_t * B - multiplicationof transposed matrix on another matrix
// outputArr.height = inpArr_A.height
__kernel void transpose_and_multiply_RELU(__global float *gradW, 
		__constant short* X,
		__constant double* dscores,
		__constant float* W,
		const float regularization,
		int num_features, int num_classes, int num_samples)
{ 
		const int row = get_global_id(0);
		const int col = get_global_id(1);
		__local float sum, elemX;
		sum = 0.;
		for(int k = 0; k < num_samples; ++k){
			elemX = (float)X[k*num_features + row];
			sum += elemX * dscores[k*num_classes + col];
		}
		gradW[num_classes*row + col] = sum + regularization * W[num_classes*row + col];
}


// performes A_t * B - multiplicationof transposed matrix on another matrix
// outputArr.height = inpArr_A.height
__kernel void transpose_and_multiply_RELU_double(__global float *gradW, 
		__constant double* X,
		__constant double* dscores,
		__constant float* W,
		const float regularization,
		int num_features, int num_classes, int num_samples)
{ 
		const int row = get_global_id(0);
		const int col = get_global_id(1);
		__local double sum;
		sum = 0.;
		for(int k = 0; k < num_samples; ++k){
			sum += X[k*num_features + row] * dscores[k*num_classes + col];
		}
		gradW[num_classes*row + col] = sum + regularization * W[num_classes*row + col];
}


// A*x + B in a matrix form
__kernel void matr_mad(__global double* Output,	// X*W + b: [num_samples x num_classes]
		__constant short* X,				// 2d array of samples
		__constant float* W,				// 2d array of weights
		__constant float* b,				// 1d array of biases
        const int width_x, const int width_w) 
{
   int i = get_global_id(0); 
   int j = get_global_id(1);
 
   // value stores the element that is computed by the thread
   __local float value, elementA, elementB;
   value = 0.;
   for (int k = 0; k < width_x; ++k) {
      elementA = (float)X[i * width_x + k];
      elementB = W[k * width_w + j];
      value = mad(elementA, elementB, value); 
   }
   Output[i * width_w + j] = value + b[j];
}

// A*x + B in a matrix form
__kernel void matr_mad_double(__global double* Output,	// X*W + b: [num_samples x num_classes]
		__global const double* X,						// 2d array of samples
		__global const float* W,						// 2d array of weights
		__global const float* b,						// 1d array of biases
        const int width_x, const int width_w) 
{
   int i = get_global_id(0); 
   int j = get_global_id(1);
   // value stores the element that is computed by the thread
   __local float value, elementA, elementB;
   value = 0.;
   for (int k = 0; k < width_x; ++k) {
      elementA = X[i * width_x + k];
      elementB = W[k * width_w + j];
      value = mad(elementA, elementB, value); 
   }
   Output[i * width_w + j] = value + b[j];
   //printf("val = %f   b = %f",value, b[j]);
}


// subtract different evement from each row
__kernel void _subtract_in_row_(__global double* inputArr, 
		__constant double* val,
		const unsigned int width)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	__local double value;
	value = inputArr[i*width + j] - val[i];
	inputArr[i*width + j] = value;
}


// exp() of every element in the matrix
__kernel void _exp_(__global double* array_A2d, 
		const unsigned int width)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	__local double exp_value;
	exp_value = exp(array_A2d[i*width + j]);
	array_A2d[i*width + j] = exp_value;
}

// computes max value in a row
__kernel void max_in_row(__constant double* inputArray, 
		__global double* max_val,
		const unsigned int width)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	__local float value;
	value = fmax(max_val[i], inputArray[i*width + j]);
	max_val[i] = value;
}


// use only if you're sertain that Output is filled with some values
// recomended use BufferFill(0) first
__kernel void sum_in_column(__constant double* array_A2d, 
		__global float* Output, const unsigned int width)
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	Output[j] += (float)array_A2d[i*width + j];
}


// calculates how many values are the same in two arrays
// doesn't work - the GPU can't compare them effectively 
__kernel  __attribute__((vec_type_hint(float)))
void compare_arr(__constant short* arr1, 
		__constant short* arr2,
		__global short* same_vals) {
	const int i = get_global_id(0);
	__local short val1, val2;
	val1 = arr1[i];
	val2 = arr2[i];
	//printf("arr1 = %d, arr2 = %d    acc  =%d",arr1[i], arr2[i], *same_vals);
	if(val1 == val2){
		same_vals[0] += 1;
	}
}

// if the number of classes is not too big, this could work faster
// if there are too many classes, need to use two kernels
__kernel __attribute__((vec_type_hint(double))) 
void softmax(__global double* probabilities, 
		const int num_classes,
		const int num_samples) 
{
	const int row = get_global_id(0);
	__local double sum, val;
	sum = 0.;
	if(row < num_samples){ 
		for(int i = 0; i < num_classes; ++i){ 
			sum += exp(probabilities[row*num_classes + i]);
		}
		sum = (sum != 0.0 ? sum : 0.000001);
		for(int i = 0; i < num_classes; ++i){ 
			val = exp(probabilities[row*num_classes + i]) / sum;
			//printf("val = %f   sum = %f,    exp_(%f) = %f", val, sum, probabilities[row*num_classes + i], exp(probabilities[row*num_classes + i]));
			probabilities[row*num_classes + i] = val;
		}
	}
}


__kernel __attribute__((vec_type_hint(double))) 
void _argmax_(__constant double* probabilities, 
		__global float* max_values,
		__global short* max_values_position,
		const int num_classes,
		const int num_samples) 
{
	const int i = get_global_id(0);
	__local double max_val_current;
	__local short position;
	position = 0;
	max_val_current = probabilities[i * num_classes];
	for(int j = 1; j < num_classes; ++j){
		if(probabilities[i * num_classes + j] > max_val_current){ 
			position = j;
			max_val_current = probabilities[i * num_classes + j];
		}
	}
	max_values[i] = max_val_current;
	max_values_position[i] = position;
	//printf("pos[%d] = %d",i, position);
}

__kernel __attribute__((vec_type_hint(double))) 
void _max_double(__constant double* val, 
		__global double* max_value) 
{
	*max_value = fmax(fabs((float)*max_value), fabs((float)val[get_global_id(0)]));
}

__kernel __attribute__((vec_type_hint(float))) 
void _max_(__constant float* val, 
		__global float* max_value) 
{
	*max_value = fmax(fabs((float)*max_value), fabs((float)val[get_global_id(0)]));
}

__kernel __attribute__((vec_type_hint(double))) 
void _norm_double_(__global double* val, 
		__constant double* max_value) 
{
	const int i = get_global_id(0);
	val[i] /= *max_value;
}

__kernel __attribute__((vec_type_hint(float))) 
void _norm_(__global float* val, 
		__constant float* max_value) 
{
	const int i = get_global_id(0);
	val[i] /= *max_value;
}