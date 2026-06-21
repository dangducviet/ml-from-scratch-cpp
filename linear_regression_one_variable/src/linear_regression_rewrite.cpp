#include<iostream>
#include<fstream>
#include<vector>
#include<cstdlib>
#include<ctime>

using namespace std;

const int epochs = 1000;

enum class LossType {
	MAE,
	MSE
};

void load_data(vector<double>& x, vector<double>& y) {
	ifstream file("dataset.txt");
	double temp_x, temp_y;

	while (file >> temp_x >> temp_y) {
		x.push_back(temp_x);
		y.push_back(temp_y);
	}
}

// prediction
double y_hat(double x, double w, double b) {
	return w * x + b;
}

//loss function: MAE
double calculate_mae(const vector<double>& x, const vector<double>& y, double w, double b) {
	int n = static_cast<int>(x.size());
	double mae = 0.0;

	for (int i = 0; i < n; i++) {
		double error = y_hat(x[i], w, b) - y[i];

		mae += (error > 0 ? error : -error);
	}
	mae /= n;

	return mae;
}

//loss function: MSE
double calculate_mse(const vector<double>& x, const vector<double>& y, double w, double b) {
	int n = static_cast<int>(x.size());
	double mse = 0.0;

	for (int i = 0; i < n; i++) {
		double error = y_hat(x[i], w, b) - y[i];

		mse += (error * error);
	}
	mse /= n;

	return mse;
}

double dw_mae(const vector<double>& x, const vector<double>& y, double w, double b) {
	int n = static_cast<int>(x.size());
	double dw_mae = 0.0;

	for (int i = 0; i < n; i++) {
		double error = y_hat(x[i], w, b) - y[i];
		double sign = (error > 0.0) ? 1.0 : ((error < 0.0) ? -1.0 : 0.0);

		dw_mae += (sign * x[i]);
	}
	dw_mae /= n; 

	return dw_mae;
}

double db_mae(const vector<double>& x, const vector<double>& y, double w, double b) {
	int n = static_cast<int>(x.size());
	double db_mae = 0.0;

	for (int i = 0; i < n; i++) {
		double error = y_hat(x[i], w, b) - y[i];
		double sign = (error > 0.0) ? 1.0 : ((error < 0.0) ? -1.0 : 0.0);

		db_mae += sign;
	}
	db_mae /= n;

	return db_mae;
}

double dw_mse(const vector<double>& x, const vector<double>& y, double w, double b) {
	int n = static_cast<int>(x.size());
	double dw_mse = 0.0;

	for (int i = 0; i < n; i++) {
		double error = y_hat(x[i], w, b) - y[i];

		dw_mse += (2 * error * x[i]);
	}
	dw_mse /= n;

	return dw_mse;
}

double db_mse(const vector<double>& x, const vector<double>& y, double w, double b) {
	int n = static_cast<int>(x.size());
	double db_mse = 0.0;

	for (int i = 0; i < n; i++) {
		double error = y_hat(x[i], w, b) - y[i];

		db_mse += (2 * error);
	}
	db_mse /= n;

	return db_mse;
}

void update_wb(double& w, double& b, double learning_rate, double dw, double db) {
	w -= (learning_rate * dw);
	b -= (learning_rate * db);
}

struct EpochRecord {
	string loss_type = "MAE";
	double learning_rate = 0.0;
	int epoch = 0;
	double w = 0.0;
	double b = 0.0;
	double loss = 0.0;
};

void save_history(vector<EpochRecord>& history, LossType loss_type, double learning_rate,
	double current_epoch, double current_w, double current_b, double current_loss) {

	int i = current_epoch + 1;
	if (i <= 5 || (i >= (epochs / 2 - 2) && i <= (epochs / 2 + 2)) || i >= (epochs - 4)) {
		EpochRecord record;

		record.loss_type = (loss_type == LossType::MAE ? "MAE" : "MSE");
		record.learning_rate = learning_rate;
		record.epoch = current_epoch + 1;
		record.w = current_w;
		record.b = current_b;
		record.loss = current_loss;

		history.push_back(record);
	}
}

void train_model(const vector<double>& x, const vector<double>& y, vector<EpochRecord>& history,
	double& w, double& b, double learning_rate, LossType loss_type) {

	double current_loss, temp_dw, temp_db;

	for (int i = 0; i < epochs; i++) {
		current_loss = (loss_type == LossType::MAE ? calculate_mae(x, y, w, b) : calculate_mse(x, y, w, b));
		temp_dw = (loss_type == LossType::MAE ? dw_mae(x, y, w, b) : dw_mse(x, y, w, b));
		temp_db = (loss_type == LossType::MAE ? db_mae(x, y, w, b) : db_mse(x, y, w, b));

		save_history(history, loss_type, learning_rate, i, w, b, current_loss);

		update_wb(w, b, learning_rate, temp_dw, temp_db);
	}
}

void summary_table(const vector<EpochRecord>& history) {
	ofstream out_file("training_history_rewrite.csv");
	out_file << "LossType,LearningRate,Epoch,w,b,Loss\n";

	for (const EpochRecord& r : history) {
		out_file << r.loss_type << ','
			<< r.learning_rate << ','
			<< r.epoch << ','
			<< r.w << ','
			<< r.b << ','
			<< r.loss << '\n';
	}

	out_file.close();

	cout << "All epoch data exported to training_history_rewrite.csv" << endl;
}

int main() {
	vector<double> x, y;
	vector<EpochRecord> history;
	double w, b;
	double learning_rate[4] = { 0.0005, 0.001, 0.005, 0.01 };
	LossType losses[] = { LossType::MAE, LossType::MSE };

	srand(time(0));
	const double initial_w = 2 * ((double)rand() / RAND_MAX) - 1;
	const double initial_b = 2 * ((double)rand() / RAND_MAX) - 1;

	load_data(x, y);

	for (LossType loss_type : losses) {
		for (int i = 0; i < 4; i++) {
			w = initial_w;
			b = initial_b;

			train_model(x, y, history, w, b, learning_rate[i], loss_type);
		}
	}
	
	summary_table(history);

	return 0;
}