#pragma once
#include "CL/cl.hpp" 
#include "CL/cl.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <random>

// assigns random values from normal distribution to an array
inline void rand_normal(float* arr, const int& size, const float max_val = 0.001) {
	srand(2000);
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(5.0, 3.0);
	int i = 0;
	while (i < size) {
		double number = distribution(generator);
		if ((number >= 0.0) && (number < max_val)) {
			arr[i] = number;
			i++;
		}
	}
}

inline void checkErrorCL(cl_int err, const char * name) noexcept {
	if (err != CL_SUCCESS) {
		std::cout << "ERROR: " << name
			<< " (" << err << ")\n" << "Press Enter to continue..." << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

inline void checkBufferCL(cl_mem& buff, const char* name) noexcept {
	if (buff == (cl_mem)0) {
		std::cout << "ERROR: Failed To Create Memory Buffer " << name
			<< "\n" << "Press Enter to exit..." << std::endl;
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}


#define GRADIENT_DESCENT 0
#define NESTEROW_DESCENT 1

// Class for logistic regression with softmax, using cross-entropy
// may be widen to use different gradient calculations, activation functions etc.
// uses openCL for optimization and multithreding
class Logistic_Regression {
public:
	Logistic_Regression() {
		 _queue_ = nullptr;
		 _context_ = nullptr;
		 _device_ = nullptr;
		_program_ = nullptr;
		platform_set = false;
		__W = nullptr;
		__b = nullptr;
		Num_FEATURES = 0; Num_CLASSES = 0;
		classificationCreated = false;
	}

	Logistic_Regression(const cl_command_queue& queue, const cl_context& context, const cl_device_id& device) {
		_queue_ = nullptr;
		_context_ = nullptr;
		_device_ = nullptr;
		_program_ = nullptr; 
		setPlatform(queue, context, device);
		set_up_program();
		__W = nullptr;
		__b = nullptr;
		Num_FEATURES = 0; Num_CLASSES = 0;
		classificationCreated = false;
	}

	~Logistic_Regression() {
		clear();
		clReleaseCommandQueue(_queue_);
		clReleaseProgram(_program_);
		clReleaseContext(_context_);
		clReleaseDevice(_device_);
	}

	inline void setPlatform(const cl_command_queue& queue, const cl_context& context, const cl_device_id& device) noexcept {
		if (platform_set) {
			clReleaseCommandQueue(_queue_);
			clReleaseContext(_context_);
			clReleaseDevice(_device_);
		}
		_queue_ = queue;
		_context_ = context;
		_device_ = device;
		set_up_program();
		platform_set = true;
	}

	// * X is excpected to be an array, representing a 2d vector, size[x_width * num_samples]
	// * y is a 1d vector of classes, size[num_samples], y[i] = (0..num_classes - 1) element
	// * learning_rate(Lipschitz constant) - prevents overflow and approximates the speed of learning 
	//		the bigger values in X, the smaller the learning_rate
	// * regularization - REGU value, helps to stabilize weights
	// * eps, max_iter - stop thresholds
	// best weights is a callback that saves the weights that has shown the best resoults during training
	inline void createClassification(short* X, short* Y,
		const int& x_width, const int& num_samples, const int& num_classes, 
		const float learning_rate = 0.01, const float regularization = 0.05,
		const float eps = 0.05, const unsigned int max_iter = 500, const bool best_weights = false) {
		if (!platform_set) {
			set_up_platform();
			set_up_program();
			platform_set = true;
		}
		reserve(x_width, num_classes);	// reserve needed memory for weights
		Optimize(X, Y, num_samples, learning_rate,
			regularization, eps, max_iter, best_weights);	// calculate weights
		classificationCreated = true;
	}
		
	// * X is excpected to be an array, representing a 2d vector, size[x_width * num_samples]
	// * y is a 1d vector of classes, size[num_samples], y[i] = (0..num_classes - 1) element
	// * learning_rate(Lipschitz constant) - prevents overflow and approximates the speed of learning 
	//		the bigger values in X, the smaller the learning_rate
	// * regularization - REGU value, helps to stabilize weights
	// * eps, max_iter - stop thresholds
	// best weights is a callback that saves the weights that has shown the best resoults during training
	inline void createClassification(double* X, short* Y,
		const int& x_width, const int& num_samples, const int& num_classes, 
		const float learning_rate = 0.01, const float regularization = 0.05,
		const float eps = 0.05, const unsigned int max_iter = 500, const bool best_weights = false) {
		if (!platform_set) {
			set_up_platform();
			set_up_program();
			platform_set = true;
		}
		reserve(x_width, num_classes);	// reserve needed memory for weights
		Optimize(X, Y, num_samples, learning_rate,
			regularization, eps, max_iter, best_weights, 1);	// calculate weights
		classificationCreated = true;
	}

	// saves weights to txt file
	// the last row conlains bias values
	int saveWeightsTXT(const char* filenameWeights) {
		std::ofstream file_W;
		file_W.open(filenameWeights);
		if (!file_W) {
			printf("Couldn't open the file");
			return -1;
		}
		for (int i = 0; i < Num_FEATURES; ++i) {
			for (int j = 0; j < Num_CLASSES - 1; ++j) {
				file_W << *(__W + i * Num_CLASSES + j) << "\t";
			}
			file_W << *(__W + i * Num_CLASSES + Num_CLASSES - 1) << "\n";
		}

		// the last row of the file will be filled with biases
		for (int j = 0; j < Num_CLASSES - 1; ++j) {
			file_W << *(__b + j) << "\t";
		}
		file_W << *(__b + Num_CLASSES - 1);

		file_W.close();
		return 0;
	}

	// reads weights fram txt file
	// * if biases is true the last row would be read as biases
	int readWeightsTXT(const char* filenameWeights, const int& num_features, 
		const int& num_clases, bool biases = true) {
		std::ifstream file_R;
		file_R.open(filenameWeights);
		if (!file_R) {
			printf("Couldn't open the file");
			return -1;
		}
		reserve(num_features, num_clases);
		for (int i = 0; i < Num_FEATURES; ++i) {
			for (int j = 0; j < Num_CLASSES; ++j)
				file_R >> *(__W + i * Num_CLASSES + j);
		}
		if (biases) {
			// the last row of the file should be filled with biases
			for (int j = 0; j < Num_CLASSES; ++j)
				file_R >> *(__b + j);
		}
		else
			memset(__b, 0., sizeof(float)*Num_CLASSES);
		file_R.close();
		classificationCreated = true;
		return 0;
	}	
	
	
	// reads just weights fram txt file
	int readJustWeightsTXT(const char* filenameWeights, 
		const int& num_features, const int& num_clases) {
		std::ifstream file_R;
		file_R.open(filenameWeights);
		if (!file_R) {
			printf("Couldn't open the file");
			return -1;
		}
		reserve(num_features, num_clases);
		for (int i = 0; i < Num_FEATURES; ++i) {
			for (int j = 0; j < Num_CLASSES; ++j)
				file_R >> *(__W + i * Num_CLASSES + j);
		}
		memset(__b, 0., sizeof(float)*Num_CLASSES);
		file_R.close();
		classificationCreated = true;
		return 0;
	}	
	
	// reads just biases fram txt file
	int readJustBiasesTXT(const char* filenameWeights, const int& num_clases) {
		std::ifstream file_R;
		file_R.open(filenameWeights);
		if (!file_R) {
			printf("Couldn't open the file");
			return -1;
		}
		delete[]__b; __b = nullptr;
		__b = new float[num_clases];
		// the last row of the file should be filled with biases
		for (int j = 0; j < Num_CLASSES; ++j){
			file_R >> *(__b + j);
		}
		file_R.close();
		return 0;
	}

	// saves weights to a binary file
	// the last row conlains bias values
	int saveWeightsBIN(const char* filenameWeights) {
		std::fstream file_W;
		file_W.open(filenameWeights, std::ios::out | std::ios::binary);
		if (!file_W) {
			printf("Couldn't open the file"); 
			return -1;
		}
		file_W.write(reinterpret_cast<char*>(__W), sizeof(*__W) * Num_CLASSES * Num_FEATURES);
		file_W.write(reinterpret_cast<char*>(__b), sizeof(*__b) * Num_CLASSES);
		file_W.close();
		return 0;
	}

	// reads weights fram binary file
	// * if biases is true the last row would be read as biases
	// * num_features - width of X array
	int readWeightsBIN(const char* filenameWeights, const int& num_features, 
		const int& num_clases, bool biases = true) {
		std::ifstream file_R;
		file_R.open(filenameWeights, std::ios::in | std::ios::binary);
		if (!file_R) {
			printf("Couldn't open the file"); 
			return -1;
		}
		reserve(num_features, num_clases);
		file_R.read(reinterpret_cast<char*>(__W), sizeof(float) * Num_CLASSES * Num_FEATURES);
		if (biases)
			file_R.read(reinterpret_cast<char*>(__b), sizeof(float) * Num_CLASSES);
		else
			memset(__b, 0., sizeof(float) * Num_CLASSES);
		file_R.close();
		classificationCreated = true;
		return 0;
	}

	// reserves memory needed for weights and biases
	void reserve(const int& num_features, const int& num_clases) {
		Num_FEATURES = num_features;
		Num_CLASSES = num_clases;
		// set up weights
		if (__W != nullptr) delete[]__W;
		if (__b != nullptr) delete[]__b;
		__W = new float[Num_FEATURES*Num_CLASSES];
		__b = new float[Num_CLASSES];
	}

	// predict the classes of objects in X
	inline int predict(short* X, short* most_likely_class, const int& num_samples) {
		if (!platform_set) {
			set_up_platform();
			set_up_program();
			platform_set = true;
		}
		int err = -1;
		if (classificationCreated) {
			err = CL_SUCCESS;
			float* max_prob = new float[num_samples];
			double* probs = new double[num_samples*Num_CLASSES];

			cl_mem X_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
				sizeof(*X) * Num_FEATURES * num_samples, X, &err);
			checkErrorCL(err, "X_buff"); checkBufferCL(X_buff, "X_buff");

			cl_mem W_buf, b_buf, likely_class_pos, likely_class, probabilities;

			__init_prediction_buffers__(W_buf, b_buf, probabilities, likely_class_pos, likely_class,
				Num_CLASSES, num_samples, Num_FEATURES, max_prob, most_likely_class, probs);
			size_t global[] = { num_samples, Num_CLASSES, 0 };
			size_t local[] = { 1, 1, 0 };

			global[0] = num_samples; global[1] = Num_CLASSES;
			err = __predict_fast__(X_buff, W_buf, b_buf, probabilities,
				likely_class, likely_class_pos, 
				Num_CLASSES, num_samples, Num_FEATURES, global, local, sizeof(*X));

			err = clEnqueueReadBuffer(_queue_, likely_class, CL_TRUE, 0,
				sizeof(*max_prob) * num_samples, max_prob, 0, NULL, NULL);
			checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer");
			err = clEnqueueReadBuffer(_queue_, likely_class_pos, CL_TRUE, 0,
				sizeof(*most_likely_class) * num_samples, most_likely_class, 0, NULL, NULL);
			checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer");

			err = clReleaseMemObject(likely_class);		checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(likely_class_pos); checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(b_buf);			checkErrorCL(err, "clReleaseMemObject : b_buf");
			err = clReleaseMemObject(W_buf);			checkErrorCL(err, "clReleaseMemObject : W_buf");
			err = clReleaseMemObject(X_buff);			checkErrorCL(err, "clReleaseMemObject : X_buf");
			delete[]max_prob;
			delete[]probs;
		}
		return err;
	}

	// predict the classes of objects in X
	inline int predict(double* X, short* most_likely_class, const int& num_samples) {
		if (!platform_set) {
			set_up_platform();
			set_up_program();
			platform_set = true;
		}
		int err = -1;
		if (classificationCreated) {
			err = CL_SUCCESS;
			float* max_prob = new float[num_samples];
			double* probs = new double[num_samples*Num_CLASSES];

			cl_mem X_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
				sizeof(*X) * Num_FEATURES * num_samples, X, &err);
			checkErrorCL(err, "X_buff"); checkBufferCL(X_buff, "X_buff");

			cl_mem W_buf, b_buf, likely_class_pos, likely_class, probabilities;

			__init_prediction_buffers__(W_buf, b_buf, probabilities, likely_class_pos, likely_class,
				Num_CLASSES, num_samples, Num_FEATURES, max_prob, most_likely_class, probs);
			size_t global[] = { num_samples, Num_CLASSES, 0 };
			size_t local[] = { 1, 1, 0 };

			global[0] = num_samples; global[1] = Num_CLASSES;
			err = __predict_fast__(X_buff, W_buf, b_buf, probabilities,
				likely_class, likely_class_pos,
				Num_CLASSES, num_samples, Num_FEATURES, global, local, 2, sizeof(*X));

			err = clEnqueueReadBuffer(_queue_, likely_class, CL_TRUE, 0,
				sizeof(*max_prob) * num_samples, max_prob, 0, NULL, NULL);
			checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer");
			err = clEnqueueReadBuffer(_queue_, likely_class_pos, CL_TRUE, 0,
				sizeof(*most_likely_class) * num_samples, most_likely_class, 0, NULL, NULL);
			checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer");

			err = clReleaseMemObject(likely_class);		checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(likely_class_pos); checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(b_buf);			checkErrorCL(err, "clReleaseMemObject : b_buf");
			err = clReleaseMemObject(W_buf);			checkErrorCL(err, "clReleaseMemObject : W_buf");
			err = clReleaseMemObject(X_buff);			checkErrorCL(err, "clReleaseMemObject : X_buf");
			delete[]max_prob;
			delete[]probs;
		}
		return err;
	}

	inline float accuracy(short* testX, short* testY, const int& num_samples) {
		if (!platform_set) {
			set_up_platform();
			set_up_program();
			platform_set = true;
		}
		int err = -1;
		int accumulator = 0;
		if (classificationCreated) {
			err = CL_SUCCESS;
			float* max_prob = new float[num_samples];
			double* probs = new double[num_samples*Num_CLASSES];
			short* most_likely_class = new short[num_samples];

			cl_mem X_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
				sizeof(*testX) * Num_FEATURES * num_samples, testX, &err);
			checkErrorCL(err, "X_buff"); checkBufferCL(X_buff, "X_buff");

			cl_mem Y_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
				sizeof(*testY) * num_samples, testY, &err);
			checkErrorCL(err, "Y_buff"); checkBufferCL(X_buff, "Y_buff");

			cl_mem W_buf, b_buf, likely_class_pos, likely_class, probabilities;

			__init_prediction_buffers__(W_buf, b_buf, probabilities, likely_class_pos, likely_class,
				Num_CLASSES, num_samples, Num_FEATURES, max_prob, most_likely_class, probs);

			size_t global[] = { num_samples, Num_CLASSES, 0 };
			size_t local[] = { 1, 1, 0 };

			global[0] = num_samples; global[1] = Num_CLASSES;
			err = __predict_fast__(X_buff, W_buf, b_buf, probabilities,
				likely_class, likely_class_pos,
				Num_CLASSES, num_samples, Num_FEATURES, global, local, 2, sizeof(*testX));

			err = clEnqueueReadBuffer(_queue_, likely_class_pos, CL_TRUE, 0,
				sizeof(*most_likely_class), most_likely_class, 0, NULL, NULL);
			checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer - loss buffer");

			for (int i = 0; i < num_samples; ++i) {
				if (most_likely_class[i] == testY[i])
					accumulator++;
			}

			err = clReleaseMemObject(likely_class);		checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(likely_class_pos); checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(b_buf);			checkErrorCL(err, "clReleaseMemObject : b_buf");
			err = clReleaseMemObject(W_buf);			checkErrorCL(err, "clReleaseMemObject : W_buf");
			err = clReleaseMemObject(X_buff);			checkErrorCL(err, "clReleaseMemObject : X_buf");
			err = clReleaseMemObject(Y_buff);			checkErrorCL(err, "clReleaseMemObject : Y_buf");
			err = clReleaseMemObject(probabilities);	checkErrorCL(err, "clReleaseMemObject : probabilities");
			delete[]max_prob;
			delete[]most_likely_class;
			delete[]probs;
		}
		return (float)accumulator / num_samples;
	}	
	
	inline float accuracy(double* testX, short* testY, const int& num_samples) {
		if (!platform_set) {
			set_up_platform();
			set_up_program();
			platform_set = true;
		}
		int err = -1; 
		int accumulator = 0;
		float acc = -1;
		if (classificationCreated) {
			err = CL_SUCCESS;
			float* max_prob = new float[num_samples];
			double* probs = new double[num_samples*Num_CLASSES];
			short* most_likely_class = new short[num_samples];

			cl_short new_val = -1;

			cl_mem X_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
				sizeof(*testX) * Num_FEATURES * num_samples, testX, &err);
			checkErrorCL(err, "X_buff"); checkBufferCL(X_buff, "X_buff");

			cl_mem Y_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
				sizeof(*testY) * num_samples, testY, &err);
			checkErrorCL(err, "Y_buff"); checkBufferCL(X_buff, "Y_buff");

			cl_mem W_buf, b_buf, likely_class_pos, likely_class, probabilities;

			__init_prediction_buffers__(W_buf, b_buf, probabilities, likely_class_pos, likely_class,
				Num_CLASSES, num_samples, Num_FEATURES, max_prob, most_likely_class, probs);

			size_t global[] = { 1, 1, 0 };
			size_t local[] = { 1, 1, 0 };

			global[0] = num_samples; 
			err = __predict_fast__(X_buff, W_buf, b_buf, probabilities,
				likely_class, likely_class_pos, Num_CLASSES, num_samples, 
				Num_FEATURES, global, local, 1, sizeof(*testX));

			//global[0] = num_samples;
			//acc = __accuracy__(Y_buff, likely_class_pos, num_samples, global, local, 1);

			err = clEnqueueReadBuffer(_queue_, likely_class_pos, CL_TRUE, 0,
				sizeof(*most_likely_class), most_likely_class, 0, NULL, NULL);
			checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer - loss buffer");

			err = clFinish(_queue_);
			checkErrorCL(err, "clFinish");

			for (int i = 0; i < num_samples; ++i) {
				if (most_likely_class[i] == testY[i])
					accumulator++;
			}

			err = clReleaseMemObject(likely_class);		checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(likely_class_pos); checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(b_buf);			checkErrorCL(err, "clReleaseMemObject : b_buf");
			err = clReleaseMemObject(W_buf);			checkErrorCL(err, "clReleaseMemObject : W_buf");
			err = clReleaseMemObject(X_buff);			checkErrorCL(err, "clReleaseMemObject : X_buf");
			err = clReleaseMemObject(Y_buff);			checkErrorCL(err, "clReleaseMemObject : Y_buf");
			err = clReleaseMemObject(probabilities);	checkErrorCL(err, "clReleaseMemObject : probabilities");
			delete[]max_prob;
			delete[]most_likely_class;
			delete[]probs;
		}
		return (float)accumulator / num_samples;
	}

	void clear() {
		if (__W != nullptr) delete[]__W;
		if (__b != nullptr) delete[]__b;
		__W = nullptr;
		__b = nullptr;
		classificationCreated = false;
		Num_FEATURES = 0; Num_CLASSES = 0;
	}

	bool isClassified() const noexcept {
		return classificationCreated;
	}

protected:
	float* __W, *__b;
	int Num_FEATURES, Num_CLASSES;
	cl_command_queue _queue_;
	cl_context _context_;
	cl_device_id _device_;
	cl_program _program_;
	bool classificationCreated;
	bool platform_set;


	// grad_alg:
	//		* 0 - gradient descent - basic unoptimized gradient descent
	//		* 1 - nesterow gradient descent - optimized gradient descent, performes better, hovewer uses more memory
	inline int Optimize(short* X, short* Y, const int& num_samples, 
			const float& learning_rate, const float& regularization,
			const float& eps, const unsigned int& max_iter,
			const bool& save_best_weights = false, const int grad_alg = GRADIENT_DESCENT) {
		int err = EXIT_SUCCESS;
		double *__scopes = new double[num_samples * Num_CLASSES];
		float *dw = new float[Num_FEATURES * Num_CLASSES];
		float *db = new float[Num_CLASSES];

		memset(__b, 0., sizeof(*__b) * Num_CLASSES);
		memset(__scopes, 1.0, sizeof(*__scopes) * num_samples * Num_CLASSES);
		memset(dw, 1.0, sizeof(*dw) * Num_FEATURES * Num_CLASSES);
		memset(db, 1.0, sizeof(*db) * Num_CLASSES);
		rand_normal(__W, Num_FEATURES * Num_CLASSES);

		cl_mem b_buf, W_buf, db_buf, dW_buf, yW_buffer, yb_buffer,
			X_buff, Y_buff, scopes_buffer, dscopes_buffer, max_vals_buf, 
			probabilities, W_previous, b_previous, likely_class_pos, likely_class;

		__init_buffers_gradient__(b_buf, W_buf, db_buf, dW_buf, 
			Y_buff, scopes_buffer, dscopes_buffer, max_vals_buf,
			Num_CLASSES, num_samples, Num_FEATURES, Y, __scopes, dw, db);

		X_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*X) * num_samples * Num_FEATURES, X, &err);
		checkErrorCL(err, "X_buff");  checkBufferCL(X_buff, "X_buff");

		// create yW, yb buffer and copy memory to them from W, b
		yW_buffer = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
			sizeof(*__W) * Num_FEATURES * Num_CLASSES, __W, &err);
		checkErrorCL(err, "yW_buffer"); checkBufferCL(yW_buffer, "yW_buffer");

		yb_buffer = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
			sizeof(*__b) * Num_CLASSES, __b, &err);
		checkErrorCL(err, "yb_buffer");  checkBufferCL(yb_buffer, "yb_buffer");

		size_t global[] = { 1, 1, 0 };
		size_t local[] = { 1, 1, 0 };

		float* max_prob = nullptr;
		short* max_prob_pos = nullptr;
		double* predicted_vals = nullptr;
		float lastAccurate = 0.0, newAccuracy = 0.0;

		if (save_best_weights) {
			max_prob = new float[num_samples];
			max_prob_pos = new short[num_samples];
			predicted_vals = new double[num_samples*Num_CLASSES];
			memset(max_prob, 0., sizeof(*max_prob) * num_samples);
			memset(max_prob_pos, -1, sizeof(*max_prob_pos) * num_samples);

			__init_prediction_buffers__(W_previous, b_previous, probabilities,
				likely_class_pos, likely_class, Num_CLASSES, num_samples, Num_FEATURES, 
				max_prob, max_prob_pos, predicted_vals);

			global[0] = num_samples; global[1] = Num_CLASSES;
			err = __predict__(X_buff, W_previous, b_previous, probabilities,
				likely_class, likely_class_pos, Num_CLASSES, num_samples, Num_FEATURES, 
				global, local, 2, sizeof(*X));

			global[0] = num_samples; global[1] = 0;
			lastAccurate = __accuracy__(Y_buff, likely_class_pos, num_samples, global, local, 1);
		}
		else {
			b_previous = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
				sizeof(*__b) * Num_CLASSES, __b, &err);
			checkErrorCL(err, "b_buf"); checkBufferCL(b_previous, "b_buf");

			W_previous = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
				sizeof(*__W) * Num_CLASSES * Num_FEATURES, __W, &err);
			checkErrorCL(err, "b_buf"); checkBufferCL(W_previous, "W_buf");
		}
		err = clFinish(_queue_);
		checkErrorCL(err, "clFinish");

		if (grad_alg == NESTEROW_DESCENT) {
			__nesterov__(X_buff, Y_buff, W_buf, b_buf, db_buf, dW_buf, yW_buffer, yb_buffer,
				scopes_buffer, dscopes_buffer, max_vals_buf, probabilities,
				W_previous, b_previous, likely_class_pos, likely_class,
				num_samples, learning_rate, regularization, eps,
				max_iter, save_best_weights, sizeof(*X));
		}
		else if (grad_alg == GRADIENT_DESCENT){
			__gradient_descent__(X_buff, Y_buff, W_buf, b_buf, db_buf, dW_buf,
				scopes_buffer, dscopes_buffer, max_vals_buf, probabilities,
				W_previous, b_previous, likely_class_pos, likely_class,
				num_samples, learning_rate, regularization, eps,
				max_iter, save_best_weights, sizeof(*X));
		}

		err = clEnqueueReadBuffer(_queue_, W_previous, CL_TRUE, 0,
			sizeof(*__W) *  Num_FEATURES * Num_CLASSES, __W, 0, NULL, NULL);
		checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer W_buf");
		err = clEnqueueReadBuffer(_queue_, b_previous, CL_TRUE, 0,
			sizeof(*__b) * Num_CLASSES, __b, 0, NULL, NULL);
		checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer b_buf");

		if (save_best_weights) {
			err = clReleaseMemObject(likely_class);		checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(likely_class_pos); checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(probabilities); checkErrorCL(err, "clReleaseMemObject : probabilities");
		}
		if (grad_alg == NESTEROW_DESCENT) {
			err = clReleaseMemObject(yW_buffer);	checkErrorCL(err, "clReleaseMemObject : yW_buffer");
			err = clReleaseMemObject(yb_buffer);	checkErrorCL(err, "clReleaseMemObject : yb_buffer");
		}
		__release_buffers__(X_buff, Y_buff, W_buf, b_buf, db_buf, dW_buf,
			scopes_buffer, dscopes_buffer, max_vals_buf, W_previous, b_previous);

		delete[]__scopes; __scopes = nullptr;
		delete[]dw; delete[]db; 
		dw = nullptr; db = nullptr;
		if(max_prob != nullptr) delete[]max_prob;
		if (max_prob_pos != nullptr) delete[]max_prob_pos;
		if (predicted_vals != nullptr) delete[]predicted_vals;
		return err;
	}

	// grad_alg:
	//		* 0 - gradient descent
	//		* 1 - nesterow gradient descent
	inline int Optimize(double* X, short* Y, const int& num_samples,
			const float& learning_rate, const float& regularization,
			const float& eps, const unsigned int& max_iter, 
			const bool& save_best_weights = false, const int grad_alg = GRADIENT_DESCENT) {
		int err = EXIT_SUCCESS;
		double *__scopes = new double[num_samples * Num_CLASSES];
		float *dw = new float[Num_FEATURES * Num_CLASSES];
		float *db = new float[Num_CLASSES];

		memset(__b, 0., sizeof(*__b) * Num_CLASSES);
		memset(__scopes, 1.0, sizeof(*__scopes) * num_samples * Num_CLASSES);
		memset(dw, 1.0, sizeof(*dw) * Num_FEATURES * Num_CLASSES);
		memset(db, 1.0, sizeof(*db) * Num_CLASSES);
		rand_normal(__W, Num_FEATURES * Num_CLASSES);

		cl_mem b_buf, W_buf, db_buf, dW_buf, yW_buffer, yb_buffer,
			X_buff, Y_buff, scopes_buffer, dscopes_buffer, max_vals_buf,
			probabilities, W_previous, b_previous, likely_class_pos, likely_class;

		__init_buffers_gradient__(b_buf, W_buf, db_buf, dW_buf,
			Y_buff, scopes_buffer, dscopes_buffer, max_vals_buf,
			Num_CLASSES, num_samples, Num_FEATURES, Y, __scopes, dw, db);

		X_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*X) * num_samples * Num_FEATURES, X, &err);
		checkErrorCL(err, "X_buff");  checkBufferCL(X_buff, "X_buff");

		if (grad_alg == NESTEROW_DESCENT) {
			// create yW, yb buffer and copy memory to them from W, b
			yW_buffer = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
				sizeof(*__W) * Num_FEATURES * Num_CLASSES, __W, &err);
			checkErrorCL(err, "yW_buffer"); checkBufferCL(yW_buffer, "yW_buffer");

			yb_buffer = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
				sizeof(*__b) * Num_CLASSES, __b, &err);
			checkErrorCL(err, "yb_buffer");  checkBufferCL(yb_buffer, "yb_buffer");
		}

		size_t global[] = { 1, 1, 0 };
		size_t local[] = { 1, 1, 0 };

		float* max_prob = nullptr;
		short* max_prob_pos = nullptr;
		double* predicted_vals = nullptr;
		float lastAccurate = 0.0, newAccuracy = 0.0;

		if (save_best_weights) {
			max_prob = new float[num_samples];
			max_prob_pos = new short[num_samples];
			predicted_vals = new double[num_samples*Num_CLASSES];
			memset(max_prob, 0., sizeof(*max_prob) * num_samples);
			memset(max_prob_pos, -1, sizeof(*max_prob_pos) * num_samples);

			__init_prediction_buffers__(W_previous, b_previous, probabilities,
				likely_class_pos, likely_class, Num_CLASSES, num_samples, Num_FEATURES,
				max_prob, max_prob_pos, predicted_vals);

			global[0] = num_samples; global[1] = Num_CLASSES;
			err = __predict__(X_buff, W_previous, b_previous, probabilities,
				likely_class, likely_class_pos, Num_CLASSES, num_samples, Num_FEATURES,
				global, local, 2, sizeof(*X));

			global[0] = num_samples; global[1] = 0;
			lastAccurate = __accuracy__(Y_buff, likely_class_pos, num_samples, global, local, 1);
		}
		else {
			b_previous = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
				sizeof(*__b) * Num_CLASSES, __b, &err);
			checkErrorCL(err, "b_buf"); checkBufferCL(b_previous, "b_buf");

			W_previous = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
				sizeof(*__W) * Num_CLASSES * Num_FEATURES, __W, &err);
			checkErrorCL(err, "b_buf"); checkBufferCL(W_previous, "W_buf");
		}
		err = clFinish(_queue_);
		checkErrorCL(err, "clFinish");
		
		if (grad_alg == NESTEROW_DESCENT) {
			__nesterov__(X_buff, Y_buff, W_buf, b_buf, db_buf, dW_buf, yW_buffer, yb_buffer,
				scopes_buffer, dscopes_buffer, max_vals_buf, probabilities,
				W_previous, b_previous, likely_class_pos, likely_class,
				num_samples, learning_rate, regularization, eps,
				max_iter, save_best_weights, sizeof(*X));
		}
		else if(grad_alg == GRADIENT_DESCENT){
			__gradient_descent__(X_buff, Y_buff, W_buf, b_buf, db_buf, dW_buf,
				scopes_buffer, dscopes_buffer, max_vals_buf, probabilities,
				W_previous, b_previous, likely_class_pos, likely_class,
				num_samples, learning_rate, regularization, eps,
				max_iter, save_best_weights, sizeof(*X));
		}

		err = clEnqueueReadBuffer(_queue_, W_previous, CL_TRUE, 0,
			sizeof(*__W) *  Num_FEATURES * Num_CLASSES, __W, 0, NULL, NULL);
		checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer W_buf");
		err = clEnqueueReadBuffer(_queue_, b_previous, CL_TRUE, 0,
			sizeof(*__b) * Num_CLASSES, __b, 0, NULL, NULL);
		checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer b_buf");

		if (save_best_weights) {
			err = clReleaseMemObject(likely_class);		checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(likely_class_pos); checkErrorCL(err, "clReleaseMemObject : loss_buff");
			err = clReleaseMemObject(probabilities); checkErrorCL(err, "clReleaseMemObject : probabilities");
		}
		if (grad_alg == NESTEROW_DESCENT) {
			err = clReleaseMemObject(yW_buffer);	checkErrorCL(err, "clReleaseMemObject : yW_buffer");
			err = clReleaseMemObject(yb_buffer);	checkErrorCL(err, "clReleaseMemObject : yb_buffer");
		}
		__release_buffers__(X_buff, Y_buff, W_buf, b_buf, db_buf, dW_buf,
			scopes_buffer, dscopes_buffer, max_vals_buf, W_previous, b_previous);

		delete[]__scopes; __scopes = nullptr;
		delete[]dw; delete[]db;
		dw = nullptr; db = nullptr;
		if (max_prob != nullptr) delete[]max_prob;
		if (max_prob_pos != nullptr) delete[]max_prob_pos;
		if (predicted_vals != nullptr) delete[]predicted_vals;
		return err;
	}

	inline void set_up_program() noexcept {
		cl_int err = CL_SUCCESS;
		std::string CLFileName = "LogisticRegression.cl";
		size_t file_size;

		char * kernel_source = read_source(CLFileName.c_str(), &file_size);

		if (NULL == kernel_source) {
			printf("Error: Failed to read kernel source code from file name: %s!\n", CLFileName.c_str());
			clReleaseContext(_context_);
			std::cin.get();
			exit(1);
		}

		_program_ = clCreateProgramWithSource(_context_, 1, (const char **)&kernel_source,
			(const size_t *)&file_size, &err);
		checkErrorCL(err, "clCreateProgramWithSource");

		printf("\nCompiling the program executable\n");


		//err = clBuildProgram(_program_, 1, &_device_, "-g -s", NULL, NULL);
		err = clBuildProgram(_program_, 1, &_device_, NULL, NULL, NULL);
		if (err != CL_SUCCESS) {
			char *buildFailre = new char[1024];
			clGetProgramBuildInfo(_program_, _device_, CL_PROGRAM_BUILD_LOG, 1024, buildFailre, NULL);
			std::cout << "Build Failure: " << buildFailre
				<< "\n Press Enter to exit the program...";
			std::cin.get();
			delete[]buildFailre;
			exit(1);
		}

		free(kernel_source);
	}

	// Choosing the platform ---> 0 - GPU
	//							  1 - CPU
	inline void set_up_platform(const int GCPU = 0) noexcept {
		std::cout << "\nSetting up OpenCL kernels\n";
		const cl_int available_platforms = 2;
		cl_int err = CL_SUCCESS;
		if (GCPU >= available_platforms)
			err = -50;
		checkErrorCL(err, "Device Id is bigger than the number of available platforms");

		// get default platform
		cl_platform_id platformList[available_platforms];
		err = clGetPlatformIDs(available_platforms, platformList, NULL);
		checkErrorCL(err, "clGetPlatformID");
		cl_platform_id default_platform = platformList[GCPU];
		char *patformName = new char[2024];
		err = clGetPlatformInfo(default_platform, CL_PLATFORM_NAME, 1024, patformName, NULL);
		checkErrorCL(err, "clGetDeviceInfo");
		std::cout << "Using platform: " << patformName << "\n";
		delete[]patformName;

		// get default device of the default platform
		cl_device_id deviceList[available_platforms];
		err = clGetDeviceIDs(default_platform, CL_DEVICE_TYPE_ALL, available_platforms, deviceList, NULL);
		checkErrorCL(err, "clGetDeviceIDs");
		if (deviceList == nullptr) {
			std::cout << " No devices found. Check OpenCL installation!\n";
			std::cin.get();
			exit(1);
		}
		_device_ = deviceList[GCPU];
		char *deviceName = new char[1024];
		err = clGetDeviceInfo(_device_, CL_DEVICE_NAME, 1024, deviceName, NULL);
		checkErrorCL(err, "clGetDeviceInfo");
		std::cout << "Using device: " << deviceName << "\n";
		delete[]deviceName;

		// platform vendor info
		char * platformVendor = new char[1024];
		err = clGetPlatformInfo(default_platform, (cl_platform_info)CL_PLATFORM_VENDOR, 1024, platformVendor, NULL);
		checkErrorCL(err, "clGetPlatformInfo");
		std::cerr << "Platform is by: " << platformVendor << "\n";
		delete[]platformVendor;

		// create context
		printf("\nCreating a compute context for the required device\n");
		_context_ = clCreateContext(NULL, 1, &_device_, NULL, NULL, &err);
		checkErrorCL(err, "clCreateContext");
		printf("\nCreating the compute program from source\n");

		_queue_ = clCreateCommandQueue(_context_, _device_, 0, &err);
		checkErrorCL(err, "clCreateCommandQueue");
	}

private:

	inline int __gradient_descent__(cl_mem& X_buff, cl_mem& Y_buff, cl_mem& W_buff, cl_mem& b_buff, 
			cl_mem& db_buf, cl_mem&  dW_buf, cl_mem& scopes_buffer, cl_mem& dscopes_buffer, cl_mem& max_vals_buf,
			cl_mem& probabilities, cl_mem& W_previous, cl_mem& b_previous, 
			cl_mem& likely_class_pos, cl_mem& likely_class, const int& num_samples,
			const float& learning_rate, const float& regularization,
			const float& eps, const unsigned int& max_iter, const bool& save_best_weights,
			const size_t x_bytesize) {
		int err = EXIT_SUCCESS;
		double loss = 1.0;
		double lastLoss = 1.;
		unsigned int iteration = 0;
		bool overflow = false;

		float lastAccurate = 0.0, newAccuracy = 0.0;
		int dim = 2;
		size_t global[] = { 1, 1, 0 };
		size_t local[] = { 1, 1, 0 };

		while (iteration < max_iter && (loss) > eps) {
			loss = get_loss_cross_entropy(X_buff, Y_buff, W_buff, b_buff, dW_buf, db_buf,
				dscopes_buffer, scopes_buffer, max_vals_buf,
				num_samples, Num_FEATURES, Num_CLASSES, regularization, x_bytesize);
			if (!std::isfinite(loss)) {
				// if there is an overflow we go back to previous weights and loss
				loss = lastLoss;
				err = clEnqueueCopyBuffer(_queue_, W_previous, W_buff, 0, 0,
					sizeof(*__W) * Num_CLASSES * Num_FEATURES, NULL, NULL, NULL);
				checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
				err = clEnqueueCopyBuffer(_queue_, b_previous, b_buff, 0, 0,
					sizeof(*__b) * Num_CLASSES, NULL, NULL, NULL);
				checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
				overflow = true;
			}


			global[0] = Num_FEATURES;
			global[1] = Num_CLASSES;
			err = __iteration_gradient_descent__(W_buff, dW_buf, Num_CLASSES,
				learning_rate, global, local, 2);

			global[0] = 1;
			global[1] = Num_CLASSES;
			err = __iteration_gradient_descent__(b_buff, db_buf, 0,
				learning_rate, global, local, 2);

			iteration += 1;

			err = clFinish(_queue_);
			checkErrorCL(err, "clFinish");

			if (save_best_weights) {
				global[0] = num_samples; global[1] = Num_CLASSES;
				err = __predict__(X_buff, W_previous, b_previous, probabilities,
					likely_class, likely_class_pos,
					Num_CLASSES, num_samples, Num_FEATURES, global, local, 2);

				global[0] = num_samples; global[1] = 0;
				newAccuracy = __accuracy__(Y_buff, likely_class_pos, num_samples, global, local, 1);

				if (newAccuracy < lastAccurate) {
					// in case previous values were better we should use previous values
					err = clEnqueueCopyBuffer(_queue_, W_previous, W_buff, 0, 0,
						sizeof(*__W) * Num_CLASSES * Num_FEATURES, NULL, NULL, NULL);
					checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
					err = clEnqueueCopyBuffer(_queue_, b_previous, b_buff, 0, 0,
						sizeof(*__b) * Num_CLASSES, NULL, NULL, NULL);
					checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
				}
				else {
					// in case new values are better we should save them in previous
					err = clEnqueueCopyBuffer(_queue_, W_buff, W_previous, 0, 0,
						sizeof(*__W) * Num_CLASSES * Num_FEATURES, NULL, NULL, NULL);
					checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy W_buf");
					err = clEnqueueCopyBuffer(_queue_, b_buff, b_previous, 0, 0,
						sizeof(*__b) * Num_CLASSES, NULL, NULL, NULL);
					checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
					lastAccurate = newAccuracy;
				}
			}
			else {
				// save previous buffer in case overflow happens
				err = clEnqueueCopyBuffer(_queue_, W_buff, W_previous, 0, 0,
					sizeof(*__W) * Num_CLASSES * Num_FEATURES, NULL, NULL, NULL);
				checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
				err = clEnqueueCopyBuffer(_queue_, b_buff, b_previous, 0, 0,
					sizeof(*__b) * Num_CLASSES, NULL, NULL, NULL);
				checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
			}

			err = clFinish(_queue_);
			checkErrorCL(err, "clFinish");
			if (!std::isfinite(loss))
				break;
			lastLoss = loss;
		}
		return err;
	}
	

	inline int __nesterov__(cl_mem& X_buff, cl_mem& Y_buff, cl_mem& W_buff, cl_mem& b_buff, 
			cl_mem& db_buf, cl_mem&  dW_buf, cl_mem& yW_buffer, cl_mem& yb_buffer,
			cl_mem& scopes_buffer, cl_mem& dscopes_buffer, cl_mem& max_vals_buf,
			cl_mem& probabilities, cl_mem& W_previous, cl_mem& b_previous, 
			cl_mem& likely_class_pos, cl_mem& likely_class, const int& num_samples,
			const float& learning_rate, const float& regularization,
			const float& eps, const unsigned int& max_iter, const bool& save_best_weights,
			const size_t x_bytesize) {
		int err = EXIT_SUCCESS;
		double loss = 1.0;
		double lastLoss = 1.;
		unsigned int iteration = 0;
		float l_prev, l, gamma;
		bool overflow = false;
		l_prev = 0.; l = 1.;

		float lastAccurate = 0.0, newAccuracy = 0.0;
		int dim = 2;
		size_t global[] = { 1, 1, 0 };
		size_t local[] = { 1, 1, 0 };

		while (iteration < max_iter && (loss) > eps) {
			loss = get_loss_cross_entropy(X_buff, Y_buff, W_buff, b_buff, dW_buf, db_buf,
				dscopes_buffer, scopes_buffer, max_vals_buf,
				num_samples, Num_FEATURES, Num_CLASSES, regularization, x_bytesize);
			if (!std::isfinite(loss)) {
				// if there is an overflow we go back to previous weights and loss
				loss = lastLoss;
				err = clEnqueueCopyBuffer(_queue_, W_previous, W_buff, 0, 0,
					sizeof(*__W) * Num_CLASSES * Num_FEATURES, NULL, NULL, NULL);
				checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
				err = clEnqueueCopyBuffer(_queue_, b_previous, b_buff, 0, 0,
					sizeof(*__b) * Num_CLASSES, NULL, NULL, NULL);
				checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
				overflow = true;
			}

			gamma = (l_prev - 1) / l;

			global[0] = Num_FEATURES;
			global[1] = Num_CLASSES;
			err = __iteration_nesterov__(W_buff, yW_buffer, dW_buf, Num_CLASSES,
				learning_rate, gamma, global, local, 2);

			global[0] = 1;
			global[1] = Num_CLASSES;
			err = __iteration_nesterov__(b_buff, yb_buffer, db_buf, 0,
				learning_rate, gamma, global, local, 2);

			l_prev = l;
			l = (1 + sqrt(1 + 4 * l_prev*l_prev))*0.5;
			iteration += 1;

			err = clFinish(_queue_);
			checkErrorCL(err, "clFinish");

			if (save_best_weights) {
				global[0] = num_samples; global[1] = Num_CLASSES;
				err = __predict__(X_buff, W_previous, b_previous, probabilities,
					likely_class, likely_class_pos,
					Num_CLASSES, num_samples, Num_FEATURES, global, local, 2);

				global[0] = num_samples; global[1] = 0;
				newAccuracy = __accuracy__(Y_buff, likely_class_pos, num_samples, global, local, 1);

				if (newAccuracy < lastAccurate) {
					// in case previous values were better we should use previous values
					err = clEnqueueCopyBuffer(_queue_, W_previous, W_buff, 0, 0,
						sizeof(*__W) * Num_CLASSES * Num_FEATURES, NULL, NULL, NULL);
					checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
					err = clEnqueueCopyBuffer(_queue_, b_previous, b_buff, 0, 0,
						sizeof(*__b) * Num_CLASSES, NULL, NULL, NULL);
					checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
				}
				else {
					// in case new values are better we should save them in previous
					err = clEnqueueCopyBuffer(_queue_, W_buff, W_previous, 0, 0,
						sizeof(*__W) * Num_CLASSES * Num_FEATURES, NULL, NULL, NULL);
					checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy W_buf");
					err = clEnqueueCopyBuffer(_queue_, b_buff, b_previous, 0, 0,
						sizeof(*__b) * Num_CLASSES, NULL, NULL, NULL);
					checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
					lastAccurate = newAccuracy;
				}
			}
			else {
				// save previous buffer in case overflow happens
				err = clEnqueueCopyBuffer(_queue_, W_buff, W_previous, 0, 0,
					sizeof(*__W) * Num_CLASSES * Num_FEATURES, NULL, NULL, NULL);
				checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
				err = clEnqueueCopyBuffer(_queue_, b_buff, b_previous, 0, 0,
					sizeof(*__b) * Num_CLASSES, NULL, NULL, NULL);
				checkErrorCL(err, "clEnqueueCopyBuffer : couldn't copy buffer");
			}

			err = clFinish(_queue_);
			checkErrorCL(err, "clFinish");
			if (!std::isfinite(loss))
				break;
			lastLoss = loss;
		}
		return err;
	}

	

	// buffer_ size : [num_samples*num_classes]
	inline double get_loss_cross_entropy(cl_mem& X, cl_mem& Y, cl_mem& W, cl_mem&b, cl_mem& dW, cl_mem&db,
			cl_mem& dscopes, cl_mem& scopes, cl_mem& max_vals,
			const int& num_samples, const int& num_features, const int& num_classes,
			const float reg, const size_t x_type_size) {
		int err = EXIT_SUCCESS;
		cl_double *loss = new cl_double[1];
		*loss = 0.;
		cl_mem loss_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*loss), loss, &err);
		checkErrorCL(err, "loss buffer");  checkBufferCL(loss_buff, "loss_buff");

		size_t global[] = { 1, 1, 0 };
		size_t local[] = { 1, 1, 0 };

		// 1) mad						
		global[0] = num_samples;
		global[1] = num_classes;
		if(x_type_size==sizeof(short))
			__mad__(W, X, b, scopes, num_classes, num_features, global, local, 2);
		else if(x_type_size==sizeof(double))
			__mad__(W, X, b, scopes, num_classes, num_features, global, local, 2, "matr_mad_double");

		// 2) normalization
		global[0] = num_samples;
		global[1] = num_classes;
		__normalize__(scopes, max_vals, num_classes, num_samples, global, local, 2);

		// 3) exp
		global[0] = num_samples;
		global[1] = num_classes;
		__exp__(scopes, num_classes, global, local, 2);

		// 4) cross-entropy with softmax activation
		global[0] = num_samples;
		global[1] = 0;
		__cross_entropy_with_softmax__(scopes, dscopes, Y, loss_buff, num_classes, num_samples, global, local, 1);

		// 5) REGU
		global[0] = num_samples;
		global[1] = num_classes;
		__regu__(W, loss_buff, reg, num_classes, global, local, 2);

		// 6) Gradient calculation: dW	// tofix
		global[0] = num_features;
		global[1] = num_classes;
		if (x_type_size == sizeof(short))
			__gradient_W__(X, dscopes, W, dW, reg, num_features, num_samples, num_classes, global, local, 2);
		else if (x_type_size == sizeof(double))
			__gradient_W__(X, dscopes, W, dW, reg, num_features, num_samples, num_classes, global, local, 2, "transpose_and_multiply_RELU_double");

		// 6) Gradient calculation: db
		global[0] = num_samples;
		global[1] = num_classes;
		__gradient_b__(dscopes, db, num_samples, num_classes, global, local, 2);

		err = clFinish(_queue_);
		checkErrorCL(err, "clFinish");

		err = clEnqueueReadBuffer(_queue_, loss_buff, CL_TRUE, 0,
			sizeof(*loss), loss, 0, NULL, NULL);
		checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer - loss buffer");

		err = clReleaseMemObject(loss_buff);
		checkErrorCL(err, "clReleaseMemObject : loss_buff");

		return *loss;
	}


	inline int __init_prediction_buffers__(cl_mem& W_buff, cl_mem& b_buff, cl_mem& probabilities,
			cl_mem& likely_class, cl_mem& likely_class_val, cl_int num_classes, 
			cl_int num_samples, cl_int num_features, 
			float* max_vals, short* probable_class, double* probs) noexcept{
		int err = CL_SUCCESS;

		b_buff = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
			sizeof(*__b) * num_classes, __b, &err);
		checkErrorCL(err, "b_buf"); checkBufferCL(b_buff, "b_buf");

		W_buff = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
			sizeof(*__W) * num_classes * num_features, __W, &err);
		checkErrorCL(err, "b_buf"); checkBufferCL(W_buff, "W_buf");

		likely_class = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*probable_class) * num_samples, probable_class, &err);
		checkErrorCL(err, "likely_class"); checkBufferCL(likely_class, "likely_class");

		likely_class_val = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*max_vals) * num_samples, max_vals, &err);
		checkErrorCL(err, "max_vals"); checkBufferCL(likely_class_val, "predictions");

		probabilities = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*probs) * num_samples * num_classes, probs, &err);
		checkErrorCL(err, "probabilities"); checkBufferCL(probabilities, "probs");

		return err;
	}

	inline int __release_buffers__(cl_mem& X_buff, cl_mem& Y_buff, 
			cl_mem& W_buff, cl_mem& b_buff, cl_mem& db_buf, cl_mem&  dW_buf,
			cl_mem& scopes_buffer, cl_mem& dscopes_buffer, cl_mem& max_vals_buf,
			cl_mem& W_previous, cl_mem& b_previous) {
		int err = CL_SUCCESS;
		err = clReleaseMemObject(b_buff);	checkErrorCL(err, "clReleaseMemObject : b_buf");
		err = clReleaseMemObject(W_buff);	checkErrorCL(err, "clReleaseMemObject : W_buf");
		err = clReleaseMemObject(X_buff);	checkErrorCL(err, "clReleaseMemObject : X_buf");
		err = clReleaseMemObject(Y_buff);	checkErrorCL(err, "clReleaseMemObject : Y_buf");
		err = clReleaseMemObject(dW_buf);	checkErrorCL(err, "clReleaseMemObject : dW_buf");
		err = clReleaseMemObject(db_buf);	checkErrorCL(err, "clReleaseMemObject : db_buf");
		err = clReleaseMemObject(b_previous);	checkErrorCL(err, "clReleaseMemObject : b_previous");
		err = clReleaseMemObject(W_previous);	checkErrorCL(err, "clReleaseMemObject : W_previous");
		err = clReleaseMemObject(scopes_buffer);	checkErrorCL(err, "clReleaseMemObject : scopes_buffer");
		err = clReleaseMemObject(dscopes_buffer);	checkErrorCL(err, "clReleaseMemObject : dscopes_buffer");
		err = clReleaseMemObject(max_vals_buf);		checkErrorCL(err, "clReleaseMemObject : max_vals_buf");
		return err;
	}

	inline int __init_buffers_gradient__(cl_mem& b_buf, cl_mem& W_buf, cl_mem& db_buf,
			cl_mem& dW_buf, cl_mem& Y_buff, cl_mem& scopes_buffer, 
			cl_mem& dscopes_buffer, cl_mem& max_vals_buf, 
			cl_int num_classes, cl_int num_samples, cl_int num_features,
			short*Y, double* __scopes, float* dw, float* db) noexcept{
		int err = CL_SUCCESS;

		Y_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*Y) * num_samples, Y, &err);
		checkErrorCL(err, "Y_buff");  checkBufferCL(Y_buff, "Y_buff");

		b_buf = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*__b) * num_classes, __b, &err);
		checkErrorCL(err, "b_buf"); checkBufferCL(b_buf, "b_buf");

		W_buf = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*__W) * num_classes * num_features, __W, &err);
		checkErrorCL(err, "b_buf"); checkBufferCL(W_buf, "W_buf");

		db_buf = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*db) * num_classes, db, &err);
		checkErrorCL(err, "db_buf"); checkBufferCL(db_buf, "db_buf");

		dW_buf = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*dw) * num_classes * num_features, dw, &err);
		checkErrorCL(err, "dW_buf"); checkBufferCL(dW_buf, "dW_buf");


		scopes_buffer = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*__scopes) * num_samples * num_classes, __scopes, &err);
		checkErrorCL(err, "scopes_buffer"); checkBufferCL(scopes_buffer, "scopesBuff");

		dscopes_buffer = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
			sizeof(*__scopes) * num_samples * num_classes, __scopes, &err);
		checkErrorCL(err, "dscopes_buffer"); checkBufferCL(dscopes_buffer, "dscopesBuff");

		max_vals_buf = clCreateBuffer(_context_, CL_MEM_COPY_HOST_PTR,
			sizeof(*__scopes) * num_samples, __scopes, &err);
		checkErrorCL(err, "max_vals_buff"); checkBufferCL(max_vals_buf, "max_vals_buf");

		return err;
	}

	inline int __predict__(cl_mem& X, cl_mem& W, cl_mem& b, cl_mem& probabilities,
			cl_mem& max_val, cl_mem& max_val_pos,
			const int& num_classes, const int& num_samples, const int& num_features,
			size_t global[3], size_t local[3], const int dims,
			const size_t x_type_size = 2) const noexcept {
		int err = CL_SUCCESS;
		// fill max buffers with smallest possible values for float
		cl_float new_val = -30000.0;
		cl_short new_val_position = -1; 

		if (x_type_size == sizeof(short))
			__mad__(W, X, b, probabilities, num_classes, num_features, global, local, 2);
		else if (x_type_size == sizeof(double))
			__mad__(W, X, b, probabilities, num_classes, num_features, global, local, 2, "matr_mad_double");

		err = __softmax__(probabilities, num_classes, num_samples, global, local, 1);

		err = clEnqueueFillBuffer(_queue_, max_val_pos, &new_val_position,
			sizeof(new_val_position), 0, sizeof(new_val_position) * num_samples, NULL, NULL, NULL);
		checkErrorCL(err, "clEnqueueFillBuffer: max_values_positions");
		err = clEnqueueFillBuffer(_queue_, max_val, &new_val,
			sizeof(new_val), 0, sizeof(new_val) * num_samples, NULL, NULL, NULL);
		checkErrorCL(err, "clEnqueueFillBuffer: max_values");	

		err = __argmax__(probabilities, num_classes, num_samples,
			max_val, max_val_pos, global, local, 2);

		return err;
	}	
	
	inline int __predict_fast__(cl_mem& X, cl_mem& W, cl_mem& b, cl_mem& probabilities,
			cl_mem& max_val, cl_mem& max_val_pos,
			const int& num_classes, const int& num_samples, const int& num_features,
			size_t global[3], size_t local[3], const int dims,
			const size_t x_type_size = 2) const noexcept {
		int err = CL_SUCCESS;
		// fill max buffers with smallest possible values for float
		cl_float new_val = -30000.0;
		cl_short new_val_position = -1; 

		if (x_type_size == sizeof(short))
			__mad__(W, X, b, probabilities, num_classes, num_features, global, local, 2);
		else if (x_type_size == sizeof(double))
			__mad__(W, X, b, probabilities, num_classes, num_features, global, local, 2, "matr_mad_double");

		err = clEnqueueFillBuffer(_queue_, max_val_pos, &new_val_position,
			sizeof(new_val_position), 0, sizeof(new_val_position) * num_samples, NULL, NULL, NULL);
		checkErrorCL(err, "clEnqueueFillBuffer: max_values_positions");
		err = clEnqueueFillBuffer(_queue_, max_val, &new_val,
			sizeof(new_val), 0, sizeof(new_val) * num_samples, NULL, NULL, NULL);
		checkErrorCL(err, "clEnqueueFillBuffer: max_values");	

		err = __argmax__(probabilities, num_classes, num_samples,
			max_val, max_val_pos, global, local, 2);

		return err;
	}

	inline int __iteration_nesterov__(cl_mem& x, cl_mem& y, cl_mem& dx, 
			const int width, const float L, const float l_n, 
			size_t global[3], size_t local[3], const int dims) const noexcept{
		int err = EXIT_SUCCESS;
		cl_kernel _grad_iteration = clCreateKernel(_program_, "nesterow_iteration", &err);
		checkErrorCL(err, "clCreateKernel - nesterow_iteration");

		err = clSetKernelArg(_grad_iteration, 0, sizeof(x), (void *)&x);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : x");

		err = clSetKernelArg(_grad_iteration, 1, sizeof(dx), (void *)&dx);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : dx");

		err = clSetKernelArg(_grad_iteration, 2, sizeof(y), (void *)&y);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : y");

		err = clSetKernelArg(_grad_iteration, 3, sizeof(width), &width);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : width");

		err = clSetKernelArg(_grad_iteration, 4, sizeof(L), &L);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : L");

		err = clSetKernelArg(_grad_iteration, 5, sizeof(l_n), &l_n);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : l_n");

		err = clEnqueueNDRangeKernel(_queue_, _grad_iteration, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : features run kernel - _grad_iteration");

		return err;
	}


	inline int __iteration_gradient_descent__(cl_mem& x, cl_mem& dx,
		const int width, const float L,
		size_t global[3], size_t local[3], const int dims) const noexcept {
		int err = EXIT_SUCCESS;
		cl_kernel _grad_iteration = clCreateKernel(_program_, "gradient_descent_iteration", &err);
		checkErrorCL(err, "clCreateKernel - gradient_descent_iteration");

		err = clSetKernelArg(_grad_iteration, 0, sizeof(x), (void *)&x);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : x");

		err = clSetKernelArg(_grad_iteration, 1, sizeof(dx), (void *)&dx);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : dx");

		err = clSetKernelArg(_grad_iteration, 2, sizeof(width), &width);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : width");

		err = clSetKernelArg(_grad_iteration, 3, sizeof(L), &L);
		checkErrorCL(err, "clSetKernelArg _grad_iteration : L");

		err = clEnqueueNDRangeKernel(_queue_, _grad_iteration, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : features run kernel - _grad_iteration");

		return err;
	}

	inline int __normalize__(cl_mem& XW_b, cl_mem& max_vals, 
			const cl_int& num_classes, const cl_int& num_samples,
			size_t global[3], size_t local[3], const int dims) const noexcept{
		int err = CL_SUCCESS;
		cl_kernel _max_ = clCreateKernel(_program_, "max_in_row", &err);
		checkErrorCL(err, "clCreateKernel - max_in_row");
		cl_kernel _subtract_ = clCreateKernel(_program_, "_subtract_in_row_", &err);
		checkErrorCL(err, "clCreateKernel - _subtract_in_row_");
		cl_double new_val = -30000.;

		err = clEnqueueFillBuffer(_queue_, max_vals, &new_val, sizeof(new_val), 
			0, sizeof(new_val)*num_samples, NULL, NULL, NULL);
		checkErrorCL(err, "clEnqueueFillBuffer");

		err = clSetKernelArg(_max_, 0, sizeof(XW_b), (void *)&XW_b);
		checkErrorCL(err, "clSetKernelArg _max_ : input array");
		err = clSetKernelArg(_max_, 1, sizeof(max_vals), (void *)&max_vals);				
		checkErrorCL(err, "clSetKernelArg _max_ : max_vals");
		err = clSetKernelArg(_max_, 2, sizeof(num_classes), &num_classes);
		checkErrorCL(err, "clSetKernelArg _max_ : num_classes");
		err = clEnqueueNDRangeKernel(_queue_, _max_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _max_");

		err = clSetKernelArg(_subtract_, 0, sizeof(XW_b), (void *)&XW_b);
		checkErrorCL(err, "clSetKernelArg _subtract_ : input array");
		err = clSetKernelArg(_subtract_, 1, sizeof(max_vals), &max_vals);
		checkErrorCL(err, "clSetKernelArg _subtract_ : max_vals");
		err = clSetKernelArg(_subtract_, 2, sizeof(num_classes), &num_classes);
		checkErrorCL(err, "clSetKernelArg _subtract_ : num_classes");
		err = clEnqueueNDRangeKernel(_queue_, _subtract_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _subtract_");

		clReleaseKernel(_max_);
		clReleaseKernel(_subtract_);
		return err;
	}

	inline int __gradient_W__(cl_mem& X, cl_mem& dscores, cl_mem& W, cl_mem& dW,
			const cl_float& reg, const cl_int& num_features, 
			const cl_int& num_samples, const cl_int& num_classes,
			size_t global[3], size_t local[3], const int& dims,
			const char* kernel_name = "transpose_and_multiply_RELU") const noexcept {
		int err = EXIT_SUCCESS;

		cl_kernel _grad_dW_ = clCreateKernel(_program_, kernel_name, &err);
		checkErrorCL(err, "clCreateKernel - transpose_and_multiply_RELU");

		err = clSetKernelArg(_grad_dW_, 0, sizeof(dW), (void *)&dW);
		checkErrorCL(err, "clSetKernelArg _grad_dW_ : dW");
		err = clSetKernelArg(_grad_dW_, 1, sizeof(X), (void *)&X);				// will be short
		checkErrorCL(err, "clSetKernelArg _grad_dW_ : X");
		err = clSetKernelArg(_grad_dW_, 2, sizeof(dscores), (void *)&dscores);
		checkErrorCL(err, "clSetKernelArg _grad_dW_ : dscores");
		err = clSetKernelArg(_grad_dW_, 3, sizeof(W), (void *)&W);
		checkErrorCL(err, "clSetKernelArg _grad_dW_ : W");
		err = clSetKernelArg(_grad_dW_, 4, sizeof(cl_float), &reg);
		checkErrorCL(err, "clSetKernelArg _grad_dW_ : reg");
		err = clSetKernelArg(_grad_dW_, 5, sizeof(cl_int), &num_features);
		checkErrorCL(err, "clSetKernelArg _grad_dW_ : num_features");
		err = clSetKernelArg(_grad_dW_, 6, sizeof(cl_int), &num_classes);
		checkErrorCL(err, "clSetKernelArg _grad_dW_ : num_classes");
		err = clSetKernelArg(_grad_dW_, 7, sizeof(cl_int), &num_samples);
		checkErrorCL(err, "clSetKernelArg _grad_dW_ : num_samples");

		err = clEnqueueNDRangeKernel(_queue_, _grad_dW_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _grad_db_");

		clReleaseKernel(_grad_dW_);
		return err;
	}

	inline int __gradient_b__(cl_mem& dscores, cl_mem& db,
			const cl_int& num_samples, const cl_int& num_classes,
			size_t global[3], size_t local[3], const int& dims) const noexcept {
		int err = EXIT_SUCCESS; 
		cl_float new_val = 0.;

		cl_kernel _grad_db_ = clCreateKernel(_program_, "sum_in_column", &err);
		checkErrorCL(err, "clCreateKernel - sum_in_column");

		err = clEnqueueFillBuffer(_queue_, db, &new_val, sizeof(new_val),
			0, sizeof(new_val)*num_classes, NULL, NULL, NULL);
		checkErrorCL(err, "clEnqueueFillBuffer");

		// calculate derevative
		err = clSetKernelArg(_grad_db_, 0, sizeof(dscores), (void *)&dscores);
		checkErrorCL(err, "clSetKernelArg _grad_db_ : dscores");
		err = clSetKernelArg(_grad_db_, 1, sizeof(db), (void *)&db);
		checkErrorCL(err, "clSetKernelArg _grad_db_ : db");
		err = clSetKernelArg(_grad_db_, 2, sizeof(cl_int), &num_classes);
		checkErrorCL(err, "clSetKernelArg _grad_db_ : num_classes");

		err = clEnqueueNDRangeKernel(_queue_, _grad_db_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _grad_db_");

		clReleaseKernel(_grad_db_);
		return err;
	}

	inline int __regu__(cl_mem& W, cl_mem& loss, const cl_float& reg, const cl_int& num_clases,
			size_t global[3], size_t local[3], const int dims) const noexcept {
		int err = EXIT_SUCCESS;
		cl_kernel _regularization_kernel_ = clCreateKernel(_program_, "regularization", &err);
		checkErrorCL(err, "clCreateKernel - regularization");

		err = clSetKernelArg(_regularization_kernel_, 0, sizeof(W), (void *)&W);
		checkErrorCL(err, "clSetKernelArg _regularization_kernel_ : output_buffer");
		err = clSetKernelArg(_regularization_kernel_, 1, sizeof(loss), (void *)&loss);
		checkErrorCL(err, "clSetKernelArg _regularization_kernel_ : output_buffer");
		err = clSetKernelArg(_regularization_kernel_, 2, sizeof(reg), &reg);
		checkErrorCL(err, "clSetKernelArg _regularization_kernel_ : output_buffer");
		err = clSetKernelArg(_regularization_kernel_, 3, sizeof(num_clases), &num_clases);
		checkErrorCL(err, "clSetKernelArg _regularization_kernel_ : output_buffer");

		err = clEnqueueNDRangeKernel(_queue_, _regularization_kernel_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _regularization_kernel_");

		clReleaseKernel(_regularization_kernel_);
		return err;

	}

	inline int __cross_entropy_with_softmax__(cl_mem& expW, cl_mem& dscores, cl_mem& y, cl_mem& loss,
			const cl_int& num_classes, const cl_int& num_samples,
			size_t global[3], size_t local[3], const int dims) const noexcept {
		int err = EXIT_SUCCESS;
		cl_kernel _softmax_kernel_ = clCreateKernel(_program_, "cross_entropy_with_softmax", &err);
		checkErrorCL(err, "clCreateKernel - cross_entropy_with_softmax");

		err = clSetKernelArg(_softmax_kernel_, 0, sizeof(expW), (void *)&expW);
		checkErrorCL(err, "clSetKernelArg _softmax_kernel_ : input_buffer");
		err = clSetKernelArg(_softmax_kernel_, 1, sizeof(dscores), (void *)&dscores);
		checkErrorCL(err, "clSetKernelArg _softmax_kernel_ : dscores");
		err = clSetKernelArg(_softmax_kernel_, 2, sizeof(y), (void *)&y);
		checkErrorCL(err, "clSetKernelArg _softmax_kernel_ : y");
		err = clSetKernelArg(_softmax_kernel_, 3, sizeof(loss), (void *)&loss);
		checkErrorCL(err, "clSetKernelArg _softmax_kernel_ : loss");
		err = clSetKernelArg(_softmax_kernel_, 4, sizeof(num_classes), &num_classes);
		checkErrorCL(err, "clSetKernelArg _softmax_kernel_ : num_classes");
		err = clSetKernelArg(_softmax_kernel_, 5, sizeof(num_samples), &num_samples);
		checkErrorCL(err, "clSetKernelArg _softmax_kernel_ : num_samples");
		err = clEnqueueNDRangeKernel(_queue_, _softmax_kernel_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _softmax_kernel_");

		clReleaseKernel(_softmax_kernel_);
		return err;

	}

	inline int __mad__(cl_mem& W, cl_mem& X, cl_mem& b, cl_mem& output_buffer,
			const cl_int& w_width, const cl_int& x_width,
			size_t global[3], size_t local[3], const int& dims, 
			const char* kernel_name = "matr_mad") const noexcept {
		int err = EXIT_SUCCESS;
		cl_kernel _mat_mad_ = clCreateKernel(_program_, kernel_name, &err);
		checkErrorCL(err, "clCreateKernel - matr_mad");

		err = clSetKernelArg(_mat_mad_, 0, sizeof(output_buffer), (void *)&output_buffer);
		checkErrorCL(err, "clSetKernelArg _mat_mad_ : output_buffer");

		err = clSetKernelArg(_mat_mad_, 1, sizeof(X), (void *)&X);		// X is short in this ocasion bacause main texture
		checkErrorCL(err, "clSetKernelArg _mat_mad_ : X");

		err = clSetKernelArg(_mat_mad_, 2, sizeof(W), (void *)&W);
		checkErrorCL(err, "clSetKernelArg _mat_mad_ : W");

		err = clSetKernelArg(_mat_mad_, 3, sizeof(b), (void *)&b);
		checkErrorCL(err, "clSetKernelArg _mat_mad_ : b");

		err = clSetKernelArg(_mat_mad_, 4, sizeof(x_width), &x_width);
		checkErrorCL(err, "clSetKernelArg _mat_mad_ : x_width");

		err = clSetKernelArg(_mat_mad_, 5, sizeof(w_width), &w_width);
		checkErrorCL(err, "clSetKernelArg _mat_mad_ : w_width");

		err = clEnqueueNDRangeKernel(_queue_, _mat_mad_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : features run kernel - _mat_mad_");

		clReleaseKernel(_mat_mad_);
		return err;
	}

	inline int __exp__(cl_mem& W, const cl_int& w_width,
			size_t global[3], size_t local[3], const int& dims) const noexcept {
		int err = EXIT_SUCCESS;
		cl_kernel _mat_exp_ = clCreateKernel(_program_, "_exp_", &err);
		checkErrorCL(err, "clCreateKernel - _exp_");

		err = clSetKernelArg(_mat_exp_, 0, sizeof(W), (void *)&W);
		checkErrorCL(err, "clSetKernelArg _mat_exp_ : input-output");
		err = clSetKernelArg(_mat_exp_, 1, sizeof(w_width), &w_width);
		checkErrorCL(err, "clSetKernelArg _mat_exp_ : width");
		err = clEnqueueNDRangeKernel(_queue_, _mat_exp_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _mat_exp_");

		clReleaseKernel(_mat_exp_);
		return err;
	}

	inline int __argmax__(cl_mem& probs,
			const cl_int& num_classes, const cl_int& num_samples,
			cl_mem& max_elem, cl_mem& max_elem_pos,
			size_t global[3], size_t local[3], const int& dims) const noexcept {
		int err = EXIT_SUCCESS;
		cl_kernel _mad_max_ = clCreateKernel(_program_, "_argmax_", &err);
		checkErrorCL(err, "clCreateKernel - mad_argmax");

		err = clSetKernelArg(_mad_max_, 0, sizeof(probs), (void *)&probs);
		checkErrorCL(err, "clSetKernelArg _mad_max_ : X");
		err = clSetKernelArg(_mad_max_, 1, sizeof(max_elem), (void *)&max_elem);
		checkErrorCL(err, "clSetKernelArg _mad_max_ : max_elem");
		err = clSetKernelArg(_mad_max_, 2, sizeof(max_elem_pos), (void *)&max_elem_pos);
		checkErrorCL(err, "clSetKernelArg _mad_max_ : max_elem_pos");
		err = clSetKernelArg(_mad_max_, 3, sizeof(num_classes), &num_classes);
		checkErrorCL(err, "clSetKernelArg _mad_max_ : x_width");
		err = clSetKernelArg(_mad_max_, 4, sizeof(num_samples), (void *)&num_samples);
		checkErrorCL(err, "clSetKernelArg _mad_max_ : w_width");

		err = clEnqueueNDRangeKernel(_queue_, _mad_max_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _mad_max_");

		clReleaseKernel(_mad_max_);
		return err;
	}

	inline float __accuracy__(cl_mem& Y, cl_mem& Y_prediction, const cl_int& num_samples,
			size_t global[3], size_t local[3], const int& dims) const noexcept {
		int err = EXIT_SUCCESS;
		cl_int* acur = new cl_int[1]{ 0 };
		cl_mem accur_buffer = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
			sizeof(*acur), acur, &err);
		checkErrorCL(err, "accuracy_buff"); checkBufferCL(accur_buffer, "accur buffer");
	
		cl_kernel _accur_ = clCreateKernel(_program_, "compare_arr", &err);
		checkErrorCL(err, "clCreateKernel - _accur_"); 
	
		err = clSetKernelArg(_accur_, 0, sizeof(Y), (void *)&Y);
		checkErrorCL(err, "clSetKernelArg _accur_ : Y");
		err = clSetKernelArg(_accur_, 1, sizeof(Y_prediction), (void *)&Y_prediction);
		checkErrorCL(err, "clSetKernelArg _accur_ : Y_pred");
		err = clSetKernelArg(_accur_, 2, sizeof(accur_buffer), (void *)&accur_buffer);
		checkErrorCL(err, "clSetKernelArg _accur_ : accur_buffer");

		global[0] = num_samples;
		err = clEnqueueNDRangeKernel(_queue_, _accur_, 1,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _accur_");
		err = clFinish(_queue_);
		checkErrorCL(err, "clFinish");
		err = clEnqueueReadBuffer(_queue_, accur_buffer, CL_TRUE, 0,
			sizeof(*acur), acur, 0, NULL, NULL);
		checkErrorCL(err, "clEnqueueReadBuffer : couldn't read from buffer - accur_buffer");
	
		err = clReleaseMemObject(accur_buffer);
		checkErrorCL(err, "clReleaseMemObject : accur_buffer");
	
		clReleaseKernel(_accur_);
		return (float)(*acur)/num_samples;
	}
	
	inline int __softmax__(cl_mem& probs,
			const cl_int& num_classes, const cl_int& num_samples,
			size_t global[3], size_t local[3], const int dims) const noexcept {
		int err = EXIT_SUCCESS;
		cl_kernel _softmax_kernel_ = clCreateKernel(_program_, "softmax", &err);
		checkErrorCL(err, "clCreateKernel - softmax");

		err = clSetKernelArg(_softmax_kernel_, 0, sizeof(probs), (void *)&probs);
		checkErrorCL(err, "clSetKernelArg _softmax_kernel_ : probs");
		err = clSetKernelArg(_softmax_kernel_, 1, sizeof(num_classes), &num_classes);
		checkErrorCL(err, "clSetKernelArg _softmax_kernel_ : num_classes");
		err = clSetKernelArg(_softmax_kernel_, 2, sizeof(num_samples), &num_samples);
		checkErrorCL(err, "clSetKernelArg _softmax_kernel_ : num_samples");
		err = clEnqueueNDRangeKernel(_queue_, _softmax_kernel_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _softmax_kernel_");

		clReleaseKernel(_softmax_kernel_);
		return err;

	}

	inline int __cross_entropy__(cl_mem& probs, cl_mem& y, cl_mem& loss,
			cl_mem& dscores, const cl_int& num_samples, const cl_int& num_classes,
			size_t global[3], size_t local[3], const int& dims) const noexcept {
		int err = EXIT_SUCCESS;
		cl_kernel _cross_entropy_kernel_ = clCreateKernel(_program_, "cross_entropy", &err);
		checkErrorCL(err, "clCreateKernel - cross_entropy");

		err = clSetKernelArg(_cross_entropy_kernel_, 0, sizeof(probs), (void *)&probs);
		checkErrorCL(err, "clSetKernelArg _cross_entropy_kernel_ : probs");
		err = clSetKernelArg(_cross_entropy_kernel_, 1, sizeof(y), (void *)&y);
		checkErrorCL(err, "clSetKernelArg _cross_entropy_kernel_ : y");
		err = clSetKernelArg(_cross_entropy_kernel_, 2, sizeof(loss), (void *)&loss);
		checkErrorCL(err, "clSetKernelArg _cross_entropy_kernel_ : loss");
		err = clSetKernelArg(_cross_entropy_kernel_, 3, sizeof(dscores), (void *)&dscores);
		checkErrorCL(err, "clSetKernelArg _cross_entropy_kernel_ : dscores");
		err = clSetKernelArg(_cross_entropy_kernel_, 4, sizeof(num_classes), &num_classes);
		checkErrorCL(err, "clSetKernelArg _cross_entropy_kernel_ : num_classes");
		err = clSetKernelArg(_cross_entropy_kernel_, 5, sizeof(num_samples), &num_samples);
		checkErrorCL(err, "clSetKernelArg _cross_entropy_kernel_ : width");

		err = clEnqueueNDRangeKernel(_queue_, _cross_entropy_kernel_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _cross_entropy_kernel_");

		clReleaseKernel(_cross_entropy_kernel_);
		return err;
	}

	inline int __norm__(cl_mem& X, size_t x_bytesize, size_t global[3],
			size_t local[3], const int& dims) const noexcept {
		int err = EXIT_SUCCESS;
		cl_mem max_buff;
		cl_kernel _kern_;
		if (x_bytesize == sizeof(cl_double)) {
			cl_double *max_val = new cl_double[1]{ -30000 };
			max_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
				sizeof(*max_val), max_val, &err);
			checkErrorCL(err, "max_buff"); checkBufferCL(max_buff, "max_buff");
			_kern_ = clCreateKernel(_program_, "_max_double", &err);
		}
		else if (x_bytesize == sizeof(cl_float)) {
			cl_float *max_val = new cl_float[1]{ -30000 };
			max_buff = clCreateBuffer(_context_, CL_MEM_USE_HOST_PTR,
				sizeof(*max_val), max_val, &err);
			checkErrorCL(err, "max_buff"); checkBufferCL(max_buff, "max_buff");
			_kern_ = clCreateKernel(_program_, "_max_", &err);
		}
		else
			return - 1;
		checkErrorCL(err, "clCreateKernel - _max_double");

		err = clSetKernelArg(_kern_, 0, sizeof(X), (void *)&X);
		checkErrorCL(err, "clSetKernelArg _mad_max_ : X");
		err = clSetKernelArg(_kern_, 1, sizeof(max_buff), (void *)&max_buff);
		checkErrorCL(err, "clSetKernelArg _mad_max_ : max_elem");

		err = clEnqueueNDRangeKernel(_queue_, _kern_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _mad_max_");

		err = clFinish(_queue_);
		checkErrorCL(err, "clFinish");

		clReleaseKernel(_kern_);

		if (x_bytesize == sizeof(cl_double))
			_kern_ = clCreateKernel(_program_, "_norm_double_", &err);
		else if (x_bytesize == sizeof(cl_float))
			_kern_ = clCreateKernel(_program_, "_norm_", &err);
		checkErrorCL(err, "clCreateKernel - _norm_double_");

		err = clSetKernelArg(_kern_, 0, sizeof(X), (void *)&X);
		checkErrorCL(err, "clSetKernelArg _mad_max_ : X");
		err = clSetKernelArg(_kern_, 1, sizeof(max_buff), (void *)&max_buff);
		checkErrorCL(err, "clSetKernelArg _mad_max_ : max_elem");

		err = clEnqueueNDRangeKernel(_queue_, _kern_, dims,
			nullptr, global, local, 0, nullptr, nullptr);
		checkErrorCL(err, "clEnqueueNDRangeKernel : kernel - _mad_max_");

		err = clFinish(_queue_);
		checkErrorCL(err, "clFinish");

		clReleaseKernel(_kern_);
		err = clReleaseMemObject(max_buff);	checkErrorCL(err, "clReleaseMemObject : max_buff");

		return err;
	}
};