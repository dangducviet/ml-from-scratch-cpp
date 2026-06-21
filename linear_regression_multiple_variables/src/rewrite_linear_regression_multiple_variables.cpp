#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

const int NUM_FEATURES = 6;
const int EPOCHS = 1000;

struct Sample {
	vector<double> x;
	double y;
};

vector<Sample> loadDataset(const string& filename) {
	vector<Sample> dataSet;
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: Could not open file" << '\n';
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
			if (!token.empty() && token.back() == '\r') {
				token.pop_back();
			}
			if (token.empty()) {
				x_temp[i] = -1.0;
			}
			else {
				x_temp[i] = stod(token);
			}
		}
		getline(ss, token, ',');
		double y_temp = stod(token);

		dataSet.push_back({ x_temp, y_temp });
	}

	return dataSet;
}

void prepareData(vector<Sample>& dataSet, vector<Sample>& trainSet, vector<Sample>& testSet, mt19937& gen) {
	shuffle(dataSet.begin(), dataSet.end(), gen);

	size_t splitData = static_cast<size_t>(dataSet.size() * 0.8);
	trainSet.assign(dataSet.begin(), dataSet.begin() + splitData);
	testSet.assign(dataSet.begin() + splitData, dataSet.end());

	cout << "Total Samples: " << dataSet.size() << '\n'
		<< "trainSet samples (80%): " << trainSet.size() << '\n'
		<< "testSet samples (20%): " << testSet.size() << '\n';
}

vector<double> calculateMedian(const vector<Sample>& trainSet) {
	vector<double> median(NUM_FEATURES);

	for (int i = 0; i < NUM_FEATURES; i++) {
		vector<double> validFeatures;
		for (const auto& sample : trainSet) {
			if (sample.x[i] != -1.0) {
				validFeatures.push_back(sample.x[i]);
			}
		}

		sort(validFeatures.begin(), validFeatures.end());

		size_t size = validFeatures.size();

		if (size % 2 == 0) {
			median[i] = (validFeatures[size / 2 - 1] + validFeatures[size / 2]) / 2;
		}
		else {
			median[i] = validFeatures[size / 2];
		}
	}

	return median;
}

void imputeMissingData(vector<Sample>& trainSet, vector<Sample>& testSet) {
	vector<double> median = calculateMedian(trainSet);

	for (int i = 0; i < NUM_FEATURES; i++) {
		for (auto& sample : trainSet) {
			if (sample.x[i] == -1.0) {
				sample.x[i] = median[i];
			}
		}

		for (auto& sample : testSet) {
			if (sample.x[i] == -1.0) {
				sample.x[i] = median[i];
			}
		}
	}
}

struct NormParams {
	vector<double> mean;
	vector<double> stddev;
};

NormParams computeNormParams(const vector<Sample>& dataset) {
	NormParams normParams;

	for (int i = 0; i < NUM_FEATURES; i++) {
		double sum = 0.0;
		for (const auto& sample : dataset) {
			sum += sample.x[i];
		}
		double mean = sum / dataset.size();
		normParams.mean.push_back(mean);

		double varianceSum = 0.0;
		for (const auto& sample : dataset) {
			double diff = sample.x[i] - mean;
			varianceSum += diff * diff;
		}
		double stddev = sqrt(varianceSum / dataset.size());
		if (stddev == 0.0) stddev = 1.0;
		normParams.stddev.push_back(stddev);
	}

	return normParams;
}

void normalizeData(vector<Sample>& trainSet, vector<Sample>& testSet, const NormParams& normParams) {
	for (int i = 0; i < NUM_FEATURES; i++) {
		for (auto& sample : trainSet) {
			sample.x[i] = (sample.x[i] - normParams.mean[i]) / normParams.stddev[i];
		}

		for (auto& sample : testSet) {
			sample.x[i] = (sample.x[i] - normParams.mean[i]) / normParams.stddev[i];
		}
	}
}

double predict(const vector<double>& x, const vector<double>& w, double b) {
	double yHat = 0.0;
	for (int i = 0; i < NUM_FEATURES; i++) {
		yHat += w[i] * x[i];
	}
	return yHat + b;
}

double computeMAE(const vector<Sample>& dataset, const vector<double>& w, const double b) {
	size_t numSamples = dataset.size();

	double sum = 0.0;
	for (const auto& sample : dataset) {
		sum += fabs(predict(sample.x, w, b) - sample.y);
	}

	return sum / numSamples;
}

double computeMSE(const vector<Sample>& dataset, const vector<double>& w, const double b) {
	size_t numSamples = dataset.size();

	double sum = 0.0;
	for (const auto& sample : dataset) {
		double diff = predict(sample.x, w, b) - sample.y;
		sum += diff * diff;
	}

	return sum / numSamples;
}

struct Gradients {
	vector<double> dw = vector<double>(NUM_FEATURES, 0.0);
	double db = 0.0;
};

Gradients computeGradientMAE(const vector<Sample>& dataset, const vector<double>& w, const double& b) {
	size_t numSamples = dataset.size();
	Gradients grad;

	for (const auto& sample : dataset) {
		double diff = predict(sample.x, w, b) - sample.y;
		double sign = (diff > 0) ? 1.0 : (diff < 0) ? -1.0 : 0.0;
		for (int i = 0; i < NUM_FEATURES; i++) {
			grad.dw[i] += sign * sample.x[i];
		}
		grad.db += sign;
	}

	double factor = 1.0 / numSamples;
	for (int i = 0; i < NUM_FEATURES; i++) {
		grad.dw[i] *= factor;
	}
	grad.db *= factor;

	return grad;
}

Gradients computeGradientMSE(const vector<Sample>& dataset, const vector<double>& w, const double& b) {
	size_t numSamples = dataset.size();
	Gradients grad;

	for (const auto& sample : dataset) {
		double diff = predict(sample.x, w, b) - sample.y;
		for (int i = 0; i < NUM_FEATURES; i++) {
			grad.dw[i] += diff * sample.x[i];
		}
		grad.db += diff;
	}
	
	double factor = 2.0 / numSamples;
	for (int i = 0; i < NUM_FEATURES; i++) {
		grad.dw[i] *= factor;
	}
	grad.db *= factor;

	return grad;
}

void updateParams(vector<double>& w, double& b, const double& learningRate, const Gradients& grad) {
	for (int i = 0; i < NUM_FEATURES; i++) {
		w[i] = w[i] - learningRate * grad.dw[i];
	}
	b = b - learningRate * grad.db;
}

struct EpochRecords {
	string lossType;
	double learningRate;
	int epoch;
	double trainLoss;
	double testLoss;
};

void summaryTable(const vector<EpochRecords>& history) {
	ofstream outfile("training_history.csv");
	if (!outfile.is_open()) {
		cerr << "Error: Could not open training_history.csv" << '\n';
	}

	outfile << "lossType,learningRate,epoch,trainLoss,testLoss\n";
	for (const auto& record : history) {
		outfile << record.lossType << ','
			<< record.learningRate << ','
			<< record.epoch << ','
			<< record.trainLoss << ','
			<< record.testLoss << '\n';
	}
}

void train(const vector<Sample>& trainSet, const vector<Sample>& testSet,
	vector<double>& w, double& b, const double& learningRate, const string& lossType,
	vector<EpochRecords>& history) {
	bool use_mae = (lossType == "MAE");

	for (int i = 0; i < EPOCHS; i++) {
		double trainLoss = use_mae ? computeMAE(trainSet, w, b) : computeMSE(trainSet, w, b);
		double testLoss = use_mae ? computeMAE(testSet, w, b) : computeMSE(testSet, w, b);

		Gradients grad = use_mae ? computeGradientMAE(trainSet, w, b) : computeGradientMSE(trainSet, w, b);
		updateParams(w, b, learningRate, grad);

		history.push_back({ lossType, learningRate, i, trainLoss, testLoss });
	}
}

void randomParams(vector<double>& initial_w, double& initial_b, mt19937& gen) {
	uniform_real_distribution<double> dist(-1.0, 1.0);
	for (int i = 0; i < NUM_FEATURES; i++) {
		initial_w[i] = dist(gen);
	}
	initial_b = dist(gen);
}

struct Best {
	vector<double> w;
	double b;
	double testLoss = numeric_limits<double>::max();
};

int main() {
	mt19937 gen(random_device{}());
	vector<Sample> dataSet = loadDataset("vietnam_housing_dataset.csv");
	vector<Sample> trainSet;
	vector<Sample> testSet;

	prepareData(dataSet, trainSet, testSet, gen);
	imputeMissingData(trainSet, testSet);
	NormParams normParams = computeNormParams(trainSet);
	normalizeData(trainSet, testSet, normParams);

	vector<double> initial_w(NUM_FEATURES);
	double initial_b;
	randomParams(initial_w, initial_b, gen);

	vector<string> lossTypes = { "MAE", "MSE" };
	vector<double> learningRate = { 0.0001, 0.001, 0.01, 0.1 };
	vector<EpochRecords> history;

	Best best;
	for (const string& lossType : lossTypes) {
		for (const double& lr : learningRate) {
			vector<double> w = initial_w;
			double b = initial_b;
			train(trainSet, testSet, w, b, lr, lossType, history);

			double finalTestLoss = history.back().testLoss;
			if (finalTestLoss < best.testLoss) {
				best.testLoss = finalTestLoss;
				best.w = w;
				best.b = b;
			}
		}
	}

	summaryTable(history);

	cout << "mean[Area]: " << normParams.mean[0] << '\n';
	cout << "stddev[Area]: " << normParams.stddev[0] << '\n';
	for (int i = 0; i < 3; i++) {
		cout << "Sample " << i << ": ";

		vector<double> x(NUM_FEATURES);
		for (int j = 0; j < NUM_FEATURES; j++) {
			cin >> x[j];
			x[j] = (x[j] - normParams.mean[j]) / normParams.stddev[j];
		}

		cout << "Predicted: " << predict(x, best.w, best.b) << '\n';
	}

	return 0;
}