#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;

const int NUM_FEATURES = 6;
const int EPOCHS = 1000;
const int NUM_PREDICTIONS = 5;

struct Sample {
	vector<double> x;
	double y;
};

vector<Sample> loadDataSet(const string& filename) {
	vector<Sample> dataSet;
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: Could not open file.\n";
		return dataSet;
	}
	string line;
	getline(file, line);

	while (getline(file, line)) {
		stringstream ss(line);
		string token;

		vector<double> x_temp(NUM_FEATURES);
		for (int i = 0; i < NUM_FEATURES; i++) {
			getline(ss, token, ',');
			x_temp[i] = stod(token);
		}
		getline(ss, token, '\n');
		double y_temp = stod(token);

		dataSet.push_back({ x_temp, y_temp });
	}

	cout << "Loaded " << dataSet.size() << " samples to dataSet.\n";
	return dataSet;
}

void splitDataSet(vector<Sample>& dataSet, vector<Sample>& trainSet, vector<Sample>& testSet, mt19937& gen) {
	shuffle(dataSet.begin(), dataSet.end(), gen);
	size_t splitIndex = static_cast<size_t>(dataSet.size() * 0.8);

	trainSet.assign(dataSet.begin(), dataSet.begin() + splitIndex);
	testSet.assign(dataSet.begin() + splitIndex, dataSet.end());

	cout << "Total samples: " << dataSet.size() << '\n'
		<< "Training samples: " << trainSet.size() << '\n'
		<< "Testing samples: " << testSet.size() << '\n';
}

struct NormParam {
	double mean;
	double stddev;
};

vector<NormParam> computeNormParams(const vector<Sample>& trainSet) {
	vector<NormParam> normParams;
	vector<double> sum(NUM_FEATURES, 0.0);
	for (const auto& sample : trainSet) {
		for (int i = 0; i < NUM_FEATURES; i++) {
			sum[i] += sample.x[i];
		}
	}
	vector<double> sumVar(NUM_FEATURES, 0.0);
	for (int i = 0; i < NUM_FEATURES; i++) {
		double mean = sum[i] / trainSet.size();
		for (const auto& sample : trainSet) {
			sumVar[i] += (sample.x[i] - mean) * (sample.x[i] - mean);
		}
		double stddev = sqrt(sumVar[i] / trainSet.size());
		if (stddev == 0.0) stddev = 1.0;
		normParams.push_back({ mean, stddev });
	}

	return normParams;
}

void normalizeData(vector<Sample>& trainSet, vector<Sample>& testSet, const vector<NormParam>& normParams) {
	for (int i = 0; i < NUM_FEATURES; i++) {
		for (auto& sample : trainSet) {
			sample.x[i] = (sample.x[i] - normParams[i].mean) / normParams[i].stddev;
		}
		for (auto& sample : testSet) {
			sample.x[i] = (sample.x[i] - normParams[i].mean) / normParams[i].stddev;
		}
	}
}

double predict(const vector<double>& x, const vector<double>& w, const double b) {
	double z = 0.0;
	for (int i = 0; i < NUM_FEATURES; i++) {
		z += x[i] * w[i];
	}
	z += b;
	return 1 / (1 + exp(-z));
}

double calculateLossWithL2Reg(const vector<Sample>& dataSet, const vector<double>& w, const double b, const double lambdaReg) {
	size_t numSamples = dataSet.size();
	double sumLoss = 0.0;
	for (const auto& sample : dataSet) {
		double p = min(max(predict(sample.x, w, b), 1e-15), 1.0 - 1e-15);
		sumLoss += (sample.y * log(p) + (1 - sample.y) * log(1 - p));
	}
	sumLoss *= -1;
	for (int i = 0; i < NUM_FEATURES; i++) {
		sumLoss += (lambdaReg / 2) * w[i] * w[i];
	}
	return sumLoss / numSamples;
}

struct Gradients {
	vector<double> dw = vector<double>(NUM_FEATURES, 0.0);
	double db = 0.0;
};

Gradients calculateGradient(const vector<Sample>& trainSet, const vector<double>& w, const double b) {
	size_t numSamples = trainSet.size();
	Gradients grad;
	
	for (const auto& sample : trainSet) {
		double diff = predict(sample.x, w, b) - sample.y;
		for (int i = 0; i < NUM_FEATURES; i++) {
			grad.dw[i] += diff * sample.x[i];
		}
		grad.db += diff;
	}

	for (int i = 0; i < NUM_FEATURES; i++) {
		grad.dw[i] /= numSamples;
	}
	grad.db /= numSamples;

	return grad;
}

void updateParamsWithL2Reg(const vector<Sample>& trainSet, vector<double>& w, double& b,
	double learningRate, double lambdaReg) {
	size_t numSamples = trainSet.size();
	Gradients grad = calculateGradient(trainSet, w, b);

	for (int i = 0; i < NUM_FEATURES; i++) {
		double regPenalty = lambdaReg * w[i] / numSamples;
		w[i] = w[i] - learningRate * (grad.dw[i] + regPenalty);
	}
	b -= learningRate * grad.db;
}

struct EpochRecords {
	double learningRate;
	double lambdaReg;
	int epoch;
	double trainLoss;
	double testLoss;
};

void summaryTable(const vector<EpochRecords>& history) {
	ofstream outfile("training_history.csv");
	if (!outfile.is_open()) {
		cerr << "Error: Could not open training_history.csv\n";
		return;
	}

	outfile << "learningRate,lambda,epoch,trainLoss,testLoss\n";
	for (const auto& record : history) {
		outfile << record.learningRate << ','
			<< record.lambdaReg << ','
			<< record.epoch << ','
			<< record.trainLoss << ','
			<< record.testLoss << '\n';
	}
}

void train(const vector<Sample>& trainSet, const vector<Sample>& testSet,
	vector<double>& w, double& b, const double learningRate, const double lambdaReg,
	vector<EpochRecords>& history) {
	for (int i = 0; i < EPOCHS; i++) {
		double trainLoss = calculateLossWithL2Reg(trainSet, w, b, lambdaReg);
		double testLoss = calculateLossWithL2Reg(testSet, w, b, lambdaReg);

		updateParamsWithL2Reg(trainSet, w, b, learningRate, lambdaReg);
		history.push_back({ learningRate, lambdaReg, i, trainLoss, testLoss });
	}
}

void randomParams(vector<double>& w, double& b, mt19937& gen) {
	uniform_real_distribution<double> dist(-1.0, 1.0);
	for (int i = 0; i < NUM_FEATURES; i++) {
		w[i] = dist(gen);
	}
	b = dist(gen);
}

struct Best {
	vector<double> w = vector<double>(NUM_FEATURES);
	double b = 0.0;
	double testLoss = numeric_limits<double>::max();
};

int main() {
	mt19937 gen(random_device{}());
	vector<Sample> dataSet = loadDataSet("breast_cancer_data.csv");
	vector<Sample> trainSet;
	vector<Sample> testSet;

	splitDataSet(dataSet, trainSet, testSet, gen);
	vector<NormParam> normParams = computeNormParams(trainSet);
	normalizeData(trainSet, testSet, normParams);

	vector<double> learningRates{ 0.001, 0.01, 0.1, 1.0, 3.0 };
	vector<double> lambdaRegs{ 0.0, 0.01, 0.1, 1.0, 10.0 };
	vector<double> initial_w(NUM_FEATURES);
	double initial_b;
	randomParams(initial_w, initial_b, gen);

	vector<EpochRecords> history;
	Best best;
	for (const auto& learningRate : learningRates) {
		for (const auto& lambdaReg : lambdaRegs) {
			vector<double> w = initial_w;
			double b = initial_b;
			train(trainSet, testSet, w, b, learningRate, lambdaReg, history);

			double finalTestLoss = history.back().testLoss;
			if (finalTestLoss < best.testLoss) {
				best.testLoss = finalTestLoss;
				best.w = w;
				best.b = b;
			}
		}
	}
	summaryTable(history);

	for (int i = 0; i < NUM_PREDICTIONS; i++) {
		cout << "Sample " << i << ": ";

		vector<double> x(NUM_FEATURES);
		for (int j = 0; j < NUM_FEATURES; j++) {
			cin >> x[j];
			x[j] = (x[j] - normParams[j].mean) / normParams[j].stddev;
		}

		cout << "Predicted: " << predict(x, best.w, best.b) << '\n';
	}

	return 0;
}


