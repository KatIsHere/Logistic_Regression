#include "Logistic_Regression"

void main() {
	short* X = new short[12]{1003, 305, 605, 2002, 4605, 6003, 5046, 2504, 2506, 4403, 5004, 100};
	short* Y = new short[3]{1, 0, 2};	
	short* X_test = new short[24]{1003, 305, 605, 2002, 4605, 6003, 5046, 2504, 3405, 
		5440, 4304, 200, 201, 4304 , 6507, 3440, 7660, 7890, 3403, 4403, 2506, 4403, 5004, 100 };
	short* Y_test = new short[6]{ 0,0,0,0,0,0 };
	Logistic_Regression classifier;
	classifier.createClassification(X, Y, 4, 3, 3, 0.00001, 0.05, 0.001, 1000, true);
	classifier.saveWeightsTXT("Classes_w.txt");
	classifier.predict(X_test, Y_test, 6);
	printf("Likelyhoods: %d, %d, %d %d, %d, %d \n ", 
		Y_test[0], Y_test[1], Y_test[2], Y_test[3], Y_test[4], Y_test[5]);
	delete[]X;
	delete[]Y;
	delete[]X_test;
	delete[]Y_test;
	printf("\nPress any key to continue...");
	std::cin.get();
}