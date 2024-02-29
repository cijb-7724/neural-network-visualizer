/*
kaggle titanic
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>

using namespace std;
using vd = vector<double>;
using vvd = vector<vector<double>>;
using vvvd = vector<vector<vector<double>>>;

typedef struct {
    vvd w;
    vvd b;
    vvd a;
    vvd x;
    vvd delta;
    vvd rw;
    vvd rb;
} layer_t;

typedef struct {
    int PassengerId;
    int Survived;
    int Pclass;
    string Name_first;
    string Name_second;
    string Sex;
    double Age;
    int SibSp;
    int Parch;
    string Ticket;
    double Fare;
    string Cabin;
    char Embarked;
} passenger_t;

//FILE
vector<passenger_t> readCSV(string filename);
vector<string> split(const string &s, char delimiter);
void missing_value(vector<passenger_t> &psg);
void normalization(vvd &x, vvd &t, vector<passenger_t> &psg);
//FUNCTION
bool judge_term(double x, double y);
vvd make_data(int n);
void make_initial_value(vvd &table, double mu, double sig);
double calc_accuracy_rate(vvd &y, vvd &t);
void shuffle_VVD(vvd &v, vector<int> &id);
//MATRIX
void matrix_show(vvd &a);
void matrix_show_b(vvd &a);
vvd matrix_multi(const vvd &a, const vvd &b);
vvd matrix_adm_multi(const vvd &a, const vvd &b);
vvd matrix_adm_multi_tensor(const vvd &a, const vvvd &b);
vvd matrix_add(const vvd &a, const vvd &b);
vvd matrix_t(const vvd &a);
//ACTIVATION
double gaussianDistribution (double mu, double sig);
double h_sigmoid(double x);
double h_tanh(double x);
double h_ReLU(double x);
vvd hm_ReLU(vvd &x);
vvd hm_tanh(vvd &x);
vvd hm_softmax(vvd &x);
double hm_cross_entropy(vvd &y, vvd &t);
//BACK PROPAGATION
vvd expansion_bias(vvd &b, int batch);
vvd calc_r_cross_entropy(vvd &x, vvd &t);
vvvd calc_r_softmax(vvd &x);
vvd calc_r_ReLU (vvd &a);
vvd calc_r_tanh(vvd &a);
vvd calc_r_bias (vvd &b, vvd &delta);
void updateWeights(vvd &w, vvd &rw, double eta);

random_device rd;
long long SEED = 0;//実行毎に同じ乱数生成
// long long SEED = rd();//実行毎に異なる
mt19937 engine(SEED);
uniform_real_distribution<> distCircle(-6, 6);

//  ##   ##    ##      ####    ##   ##
//  ### ###   ####      ##     ###  ##
//  #######  ##  ##     ##     #### ##
//  #######  ##  ##     ##     ## ####
//  ## # ##  ######     ##     ##  ###
//  ##   ##  ##  ##     ##     ##   ##
//  ##   ##  ##  ##    ####    ##   ##

/*

*/
vector<passenger_t> readCSV_test(string filename);
void normalization_test(vvd &x, vector<passenger_t> &psg);
void test_output(vector<layer_t> &nn, int depth);


int main() {
    string filename = "data/train.csv";
    vector<passenger_t> passengers = readCSV(filename);

    cout << "size is ";
    cout << passengers.size() << endl;
    vvd data_x, data_t;
    missing_value(passengers);
    normalization(data_x, data_t, passengers);

    cout << "size is ";
    cout << passengers.size() << endl;

    int data_size = data_x.size();
    int train_size = data_size * 0.7;
    int valid_size = data_size - train_size;
    vvd train_x, train_t, valid_x, valid_t;
    
    vector<int> id(data_size);
    for (int i=0; i<data_size; ++i) id[i] = i;
    shuffle(id.begin(), id.end(), engine);
    for (int i=0; i<data_size; ++i) {
        if (i < train_size) {
            train_x.push_back(data_x[i]);
            train_t.push_back(data_t[i]);
        } else {
            valid_x.push_back(data_x[i]);
            valid_t.push_back(data_t[i]);
        }
    }
    
    double eta = 0.03, attenuation = 0.1;
    int show_interval = 100;
    int learning_plan = 100;
    int loop = 500;
    int batch_size = 400; //<train_size
    // vector<int> nn_form = {6, 100, 100, 2};
    vector<int> nn_form = {7, 10, 10, 10, 10, 10, 10, 2};
    
    int depth = nn_form.size()-1;

    vector<layer_t> nn(depth);

    //Heの初期化
    for (int i=0; i<depth; ++i) {
        nn[i].w.assign(nn_form[i], vd(nn_form[i+1], 0));
        nn[i].b.assign(1, vd(nn_form[i+1], 0));
        make_initial_value(nn[i].w, 0, sqrt(2.0/nn_form[i]));
        make_initial_value(nn[i].b, 0, sqrt(2.0/nn_form[i]));
        nn[i].b = expansion_bias(nn[i].b, batch_size);
    }
    id.assign(train_size, 0);
    for (int i=0; i<train_size; ++i) id[i] = i;

    
    //learn
    for (int i=0; i<loop; ++i) {
        // cout << i << endl;
        //mini batchの作成
        shuffle(id.begin(), id.end(), engine);
        shuffle_VVD(train_x, id);
        shuffle_VVD(train_t, id);
        
        //全データから先頭batchi_sizeだけmini batchを取得
        vvd x0, t0;
        for (int j=0; j<batch_size; ++j) {
            x0.push_back(train_x[j]);
            t0.push_back(train_t[j]);
        }

        //forward propagation
        for (int k=0; k<depth; ++k) {
            if (k == 0) nn[k].a = matrix_add(matrix_multi(x0, nn[k].w), nn[k].b);
            else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
            // if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
            if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
            else nn[k].x = hm_softmax(nn[k].a);
        }

        //back propagation
        for (int k=depth-1; k>=0; --k) {
            if (k == depth-1) {
                vvd r_fL_xk;
                vvvd r_hk_ak;
                r_fL_xk = calc_r_cross_entropy(nn[k].x, t0);
                r_hk_ak = calc_r_softmax(nn[k].x);
                nn[k].delta = matrix_adm_multi_tensor(r_fL_xk, r_hk_ak);
            } else {
                vvd r_h_a;
                // r_h_a = calc_r_ReLU(nn[k].a);
                r_h_a = calc_r_tanh(nn[k].a);
                nn[k].delta = matrix_adm_multi(r_h_a, matrix_multi(nn[k+1].delta, matrix_t(nn[k+1].w)));
            }
            nn[k].rb = calc_r_bias(nn[k].b, nn[k].delta);
            if (k != 0) nn[k].rw = matrix_multi(matrix_t(nn[k-1].x), nn[k].delta);
            else nn[k].rw = matrix_multi(matrix_t(x0), nn[k].delta);
        }

        //update parameters
        for (int k=0; k<depth; ++k) {
            updateWeights(nn[k].w, nn[k].rw, eta);
            updateWeights(nn[k].b, nn[k].rb, eta);
        }
        //学習率の更新
        if ((i+1) % learning_plan == 0) eta *= attenuation;
        // if ((i+1) % learning_plan*10 == 0) eta = 0.01;

        //たまに性能の確認
        if (i % show_interval == 0) {
            cout << i << " cross entropy ";
            cout << hm_cross_entropy(nn[depth-1].x, t0) << endl;
            cout << "accuracy rate ";
            cout << calc_accuracy_rate(nn[depth-1].x, t0) << endl;
        }
    }

    // train set---------------------------------
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;
    cout << "train set" << endl;
    for (int i=0; i<depth; ++i) {
        nn[i].b = expansion_bias(nn[i].b, train_size);
    }
    //forward propagation
    for (int k=0; k<depth; ++k) {
        if (k == 0) nn[k].a = matrix_add(matrix_multi(train_x, nn[k].w), nn[k].b);
        else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
        // if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
        if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
        else nn[k].x = hm_softmax(nn[k].a);
    }
    cout << " cross entropy ";
    cout << hm_cross_entropy(nn[depth-1].x, train_t) << endl;
    cout << "accuracy rate ";
    cout << calc_accuracy_rate(nn[depth-1].x, train_t) << endl;
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;

    // valid set-------------------------------------
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;
    cout << "valid set" << endl;
    for (int i=0; i<depth; ++i) {
        nn[i].b = expansion_bias(nn[i].b, valid_size);
    }

    //forward propagation
    for (int k=0; k<depth; ++k) {
        if (k == 0) nn[k].a = matrix_add(matrix_multi(valid_x, nn[k].w), nn[k].b);
        else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
        // if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
        if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
        else nn[k].x = hm_softmax(nn[k].a);
    }
    cout << " cross entropy ";
    cout << hm_cross_entropy(nn[depth-1].x, valid_t) << endl;
    cout << "accuracy rate ";
    cout << calc_accuracy_rate(nn[depth-1].x, valid_t) << endl;
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;

    //testset 
    test_output(nn, depth);
}
//  #######   ####    ####     #######
//   ##   #    ##      ##       ##   #
//   ## #      ##      ##       ## #
//   ####      ##      ##       ####
//   ## #      ##      ##   #   ## #
//   ##        ##      ##  ##   ##   #
//  ####      ####    #######  #######

/*
PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
*/
vector<passenger_t> readCSV(string filename) {
    vector<passenger_t> passengers;
    ifstream file(filename);

    if (!file.is_open()) {
        cout << "File not found." << endl;
        return passengers;
    }

    string line;
    getline(file, line); // ヘッダー行を読み飛ばす
    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        passenger_t passenger;

        passenger.PassengerId = stoi(tokens[0]);
        passenger.Survived = stoi(tokens[1]);
        passenger.Pclass = stoi(tokens[2]);
        passenger.Name_first = tokens[3];
        passenger.Name_second = tokens[4];
        passenger.Sex = tokens[5];
        passenger.Age = tokens[6].empty() ? -1 : stoi(tokens[6]);
        passenger.SibSp = tokens[7].empty() ? -1 : stoi(tokens[7]);
        passenger.Parch = tokens[8].empty() ? -1 : stoi(tokens[8]);
        passenger.Ticket = tokens[9];
        passenger.Fare = tokens[10].empty() ? -1 : stod(tokens[10]);
        passenger.Cabin = tokens[11];
        passenger.Embarked = tokens[12].empty() ? ' ' : tokens[12][0];

        passengers.push_back(passenger);//欠損値は-1 or space
    }

    file.close();
    return passengers;
}
vector<passenger_t> readCSV_test(string filename) {
    vector<passenger_t> passengers;
    ifstream file(filename);

    if (!file.is_open()) {
        cout << "File not found." << endl;
        return passengers;
    }

    string line;
    getline(file, line); // ヘッダー行を読み飛ばす
    while (getline(file, line)) {
        vector<string> tokens = split(line, ',');
        passenger_t passenger;

        passenger.PassengerId = stoi(tokens[0]);
        // passenger.Survived = stoi(tokens[1]);
        passenger.Pclass = stoi(tokens[1]);
        passenger.Name_first = tokens[2];
        passenger.Name_second = tokens[3];
        passenger.Sex = tokens[4];
        passenger.Age = tokens[5].empty() ? -1 : stoi(tokens[5]);
        passenger.SibSp = tokens[6].empty() ? -1 : stoi(tokens[6]);
        passenger.Parch = tokens[7].empty() ? -1 : stoi(tokens[7]);
        passenger.Ticket = tokens[8];
        passenger.Fare = tokens[9].empty() ? -1 : stod(tokens[9]);
        passenger.Cabin = tokens[10];
        passenger.Embarked = tokens[11].empty() ? ' ' : tokens[11][0];

        passengers.push_back(passenger);//欠損値は-1 or space
    }

    file.close();
    return passengers;
}

vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    stringstream ss(s);
    string token;
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}
//medianで補完
//age, fareのみ
void missing_value(vector<passenger_t> &psg) {
    int n = psg.size();
    vector<int> age;
    for (int i=0; i<n; ++i) {
        if (psg[i].Age != -1) age.push_back(psg[i].Age);
    }
    sort(age.begin(), age.end());
    for (int i=0; i<n; ++i) {
        if (psg[i].Age == -1) psg[i].Age = age[age.size()/2];
    }
    vd fare;
    for (int i=0; i<n; ++i) {
        if (psg[i].Fare != -1) fare.push_back(psg[i].Fare);
    }
    sort(fare.begin(), fare.end());
    for (int i=0; i<n; ++i) {
        if (psg[i].Fare == -1) psg[i].Fare = fare[fare.size()/2];
    }

    vector<char> embarked;
    for (int i=0; i<n; ++i) {
        if (psg[i].Embarked == ' ') psg[i].Embarked = 'S';
    }
}

void normalization(vvd &x, vvd &t, vector<passenger_t> &psg) {
    x.assign(0, vd(0));
    t.assign(0, vd(0));
    int n = psg.size();
    //正規化と特徴量エンジニアリング
    /*
    Pclass:金持ち権力出ちゃうかも　いる！
    Name:名前は確認しないだろう　いらない！
    Sex:これはありそう　いる！
    Age:丈夫な体の人は寒くても生き残るだろう　いる！
    SibSp:とっさにそこを考慮して優先してボートに乗せるか？　いらない！add
    Parch:とっさにそこを考慮して優先してボートに乗せるか？　いらない！add
    Ticket:チケットも確認されないだろう　いらない！
    Fare:金持ち権力出ちゃうかも　いる！
    Cabin:部屋も金持ちと相関あるだろうけど，欠損値が多すぎるから捨てる　いらない！
    Embarked:港も今回は無視　いらない！
    Pclass, Sex, Age, Fare
    */
    vd vec_Pclass, vec_Sex, vec_Age, vec_Fare;
    vd vec_SibSp, vec_Parch, vec_Emb;
    for (int i=0; i<n; ++i) {
        vec_Pclass.push_back(psg[i].Pclass);
        if (psg[i].Sex == "male") vec_Sex.push_back(0);
        else vec_Sex.push_back(1);
        vec_Age.push_back(psg[i].Age);
        vec_Fare.push_back(psg[i].Fare);
        vec_SibSp.push_back(psg[i].SibSp);
        vec_Parch.push_back(psg[i].Parch);
        if (psg[i].Embarked == 'S') vec_Emb.push_back(1);
        else if (psg[i].Embarked == 'C') vec_Emb.push_back(0);
        else vec_Emb.push_back(-1);
    }
    double mx_Pclass, mn_Pclass, mx_Age, mn_Age, mx_Fare, mn_Fare;
    double mx_SibSp, mn_SibSp, mx_Parch, mn_Parch;
    mx_Pclass = *max_element(vec_Pclass.begin(), vec_Pclass.end());
    mn_Pclass = *min_element(vec_Pclass.begin(), vec_Pclass.end());
    mx_Age = *max_element(vec_Age.begin(), vec_Age.end());
    mn_Age = *min_element(vec_Age.begin(), vec_Age.end());
    mx_Fare = *max_element(vec_Fare.begin(), vec_Fare.end());
    mn_Fare = *min_element(vec_Fare.begin(), vec_Fare.end());
    mx_SibSp = *max_element(vec_SibSp.begin(), vec_SibSp.end());
    mn_SibSp = *min_element(vec_SibSp.begin(), vec_SibSp.end());
    mx_Parch = *max_element(vec_Parch.begin(), vec_Parch.end());
    mn_Parch = *min_element(vec_Parch.begin(), vec_Parch.end());

    for (int i=0; i<n; ++i) {
        vec_Pclass[i] = (vec_Pclass[i] - mn_Pclass) / (mx_Pclass - mn_Pclass);
        vec_Age[i] = (vec_Age[i] - mn_Age) / (mx_Age - mn_Age);
        vec_Fare[i] = (vec_Fare[i] - mn_Fare) / (mx_Fare - mn_Fare);
        vec_SibSp[i] = (vec_SibSp[i] - mn_SibSp) / (mx_SibSp - mn_SibSp);
        vec_Parch[i] = (vec_Parch[i] - mn_Parch) / (mx_Parch - mn_Parch);
    }

    for (int i=0; i<n; ++i) {
        //教師ラベル
        if (psg[i].Survived == 1) t.push_back({1, 0});
        else t.push_back({0, 1});
        //訓練インスタンス
        x.push_back({vec_Pclass[i], vec_Sex[i], vec_Age[i], vec_Fare[i], vec_SibSp[i], vec_Parch[i], vec_Emb[i]});
    }
}

void normalization_test(vvd &x, vector<passenger_t> &psg) {
    x.assign(0, vd(0));
    int n = psg.size();
    vd vec_Pclass, vec_Sex, vec_Age, vec_Fare;
    vd vec_SibSp, vec_Parch, vec_Emb;
    for (int i=0; i<n; ++i) {
        vec_Pclass.push_back(psg[i].Pclass);
        if (psg[i].Sex == "male") vec_Sex.push_back(0);
        else vec_Sex.push_back(1);
        vec_Age.push_back(psg[i].Age);
        vec_Fare.push_back(psg[i].Fare);
        vec_SibSp.push_back(psg[i].SibSp);
        vec_Parch.push_back(psg[i].Parch);
        if (psg[i].Embarked == 'S') vec_Emb.push_back(1);
        else if (psg[i].Embarked == 'C') vec_Emb.push_back(0);
        else vec_Emb.push_back(-1);
    }

    double mx_Pclass, mn_Pclass, mx_Age, mn_Age, mx_Fare, mn_Fare;
    double mx_SibSp, mn_SibSp, mx_Parch, mn_Parch;
    mx_Pclass = *max_element(vec_Pclass.begin(), vec_Pclass.end());
    mn_Pclass = *min_element(vec_Pclass.begin(), vec_Pclass.end());
    mx_Age = *max_element(vec_Age.begin(), vec_Age.end());
    mn_Age = *min_element(vec_Age.begin(), vec_Age.end());
    mx_Fare = *max_element(vec_Fare.begin(), vec_Fare.end());
    mn_Fare = *min_element(vec_Fare.begin(), vec_Fare.end());
    mx_SibSp = *max_element(vec_SibSp.begin(), vec_SibSp.end());
    mn_SibSp = *min_element(vec_SibSp.begin(), vec_SibSp.end());
    mx_Parch = *max_element(vec_Parch.begin(), vec_Parch.end());
    mn_Parch = *min_element(vec_Parch.begin(), vec_Parch.end());

    for (int i=0; i<n; ++i) {
        vec_Pclass[i] = (vec_Pclass[i] - mn_Pclass) / (mx_Pclass - mn_Pclass);
        vec_Age[i] = (vec_Age[i] - mn_Age) / (mx_Age - mn_Age);
        vec_Fare[i] = (vec_Fare[i] - mn_Fare) / (mx_Fare - mn_Fare);
        vec_SibSp[i] = (vec_SibSp[i] - mn_SibSp) / (mx_SibSp - mn_SibSp);
        vec_Parch[i] = (vec_Parch[i] - mn_Parch) / (mx_Parch - mn_Parch);
    }

    for (int i=0; i<n; ++i) {
        //訓練インスタンス
        x.push_back({vec_Pclass[i], vec_Sex[i], vec_Age[i], vec_Fare[i], vec_SibSp[i], vec_Parch[i], vec_Emb[i]});
    }
}
void test_output(vector<layer_t> &nn, int depth) {
    string filename = "data/test.csv";
    vector<passenger_t> passengers = readCSV_test(filename);
    cout << "after read csv" << endl;
    vvd test_x;
    missing_value(passengers);
    normalization_test(test_x, passengers);
    int test_size = test_x.size();

    // valid set-------------------------------------
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;
    cout << "test set" << endl;
    for (int i=0; i<depth; ++i) {
        nn[i].b = expansion_bias(nn[i].b, test_size);
    }

    //forward propagation
    for (int k=0; k<depth; ++k) {
        if (k == 0) nn[k].a = matrix_add(matrix_multi(test_x, nn[k].w), nn[k].b);
        else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
        // if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
        if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
        else nn[k].x = hm_softmax(nn[k].a);
    }
    //nn[depth-1].x
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;

    // matrix_show(nn[depth-1].x);
    for (int i=0; i<5; ++i) {
        cout << passengers[i].PassengerId << "," << nn[depth-1].x[i][0] << ' ' << nn[depth-1].x[i][1] << endl;
    }

    filename = "data/submission.csv";
    ofstream outputFile (filename);
    outputFile << "PassengerId,Survived" << endl;
    for (int i=0; i<test_size; ++i) {
        if (nn[depth-1].x[i][0] > nn[depth-1].x[i][1]) {
            outputFile << passengers[i].PassengerId << ",1" << endl;
        } else {
            outputFile << passengers[i].PassengerId << ",0" << endl;
        }   
    }
    
}
//  #######  ##   ##  ##   ##    ####   ######    ####     #####   ##   ##
//   ##   #  ##   ##  ###  ##   ##  ##  # ## #     ##     ##   ##  ###  ##
//   ## #    ##   ##  #### ##  ##         ##       ##     ##   ##  #### ##
//   ####    ##   ##  ## ####  ##         ##       ##     ##   ##  ## ####
//   ## #    ##   ##  ##  ###  ##         ##       ##     ##   ##  ##  ###
//   ##      ##   ##  ##   ##   ##  ##    ##       ##     ##   ##  ##   ##
//  ####      #####   ##   ##    ####    ####     ####     #####   ##   ##


void make_initial_value(vvd &table, double mu, double sig) {
    int n = table.size(), m = table[0].size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            table[i][j] = gaussianDistribution(mu, sig);
        }
    }
}


double calc_accuracy_rate(vvd &y, vvd &t) {
    int n = y.size(), m = y[0].size();
    double sum = 0;
    for (int i=0; i<n; ++i) {
        double mx = *max_element(y[i].begin(), y[i].end());
        for (int j=0; j<m; ++j) {
            if (y[i][j] == mx && t[i][j]) sum += 1;
        }
    }
    return sum / n;
}

void shuffle_VVD(vvd &v, vector<int> &id) {
    vvd tmp = v;
    int n = v.size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<v[0].size(); ++j) {
            tmp[i][j] = v[id[i]][j];
        }
    }
    v = tmp;
}


//  ##   ##    ##     ######   ######    ####    ##  ##
//  ### ###   ####    # ## #    ##  ##    ##     ##  ##
//  #######  ##  ##     ##      ##  ##    ##      ####
//  #######  ##  ##     ##      #####     ##       ##
//  ## # ##  ######     ##      ## ##     ##      ####
//  ##   ##  ##  ##     ##      ##  ##    ##     ##  ##
//  ##   ##  ##  ##    ####    #### ##   ####    ##  ##

// show
void matrix_show(vvd &a) {
    int n = a.size(), m = a[0].size();
    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            cout << a[i][j] << ' ';
        }
        cout << endl;
    }
    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;
}
// biasは各行同じものがバッチサイズ分あるので最初の1行だけ表示
void matrix_show_b(vvd &a) {
    int m = a[0].size();
    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;

    for (int j=0; j<m; ++j) cout << a[0][j] << ' ';
    cout << endl;

    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;
}

// c = a * b
vvd matrix_multi(const vvd &a, const vvd &b) {
    int n = a.size(), m = b.size(), l = b[0].size();
    vvd c;
    c.assign(n, vd(l, 0));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<l; ++j) {
            for (int k=0; k<m; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}
// c = a .* b (admal)
vvd matrix_adm_multi(const vvd &a, const vvd &b) {
    int n = a.size(), m = a[0].size();
    vvd c;
    c.assign(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            c[i][j] = a[i][j] * b[i][j];
        }
    }
    return c;
}

// c = a + b
vvd matrix_add(const vvd &a, const vvd &b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        cout << "The matrix sizes are different." << endl;
        vvd ret = {{0}};
        return ret;
    }
    int n = a.size(), m = a[0].size();
    vvd c;
    c.assign(n, vd(m, 0));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            c[i][j] += a[i][j] + b[i][j];
        }
    }
    return c;
}

// a = a^T
vvd matrix_t(const vvd &a) {
    int n = a.size(), m = a[0].size();
    vvd t;
    t.assign(m, vd(n));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            t[j][i] = a[i][j];
        }
    }
    return t;
}
// c = a .* b, a:matrix, b:tensor
vvd matrix_adm_multi_tensor(const vvd &a, const vvvd &b) {
    if (a.size() != b.size()) {
        cout << "The matrix sizes are different. 1dimention" << endl;
        return {{}};
    }
    if (a[0].size() != b[0].size()) {
        cout << "The matrix sizes are different. 2dimention" << endl;
    }
    int n = a.size(), m = a[0].size();
    vvd ret(n, vd(m, 0));
    for (int i=0; i<n; ++i) {
        vvd tmp = {a[i]};
        tmp = matrix_multi(tmp, b[i]);
        ret[i] = tmp[0];
    }
    return ret;
}


//    ##       ####   ######    ####    ##   ##    ##     ######    ####     #####   ##   ##
//   ####     ##  ##  # ## #     ##     ##   ##   ####    # ## #     ##     ##   ##  ###  ##
//  ##  ##   ##         ##       ##      ## ##   ##  ##     ##       ##     ##   ##  #### ##
//  ##  ##   ##         ##       ##      ## ##   ##  ##     ##       ##     ##   ##  ## ####
//  ######   ##         ##       ##       ###    ######     ##       ##     ##   ##  ##  ###
//  ##  ##    ##  ##    ##       ##       ###    ##  ##     ##       ##     ##   ##  ##   ##
//  ##  ##     ####    ####     ####       #     ##  ##    ####     ####     #####   ##   ##

double gaussianDistribution (double mu, double sig) {
    normal_distribution <> dist(mu, sig);
    return dist(engine);
}
double h_sigmoid(double x) {
    return 1/(1+exp(-x));
}
double h_tanh(double x) {
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x));
}
double h_ReLU(double x) {
    return (x > 0) ? x : 0;
}

vvd hm_ReLU(vvd &x) {
    int n = x.size(), m = x[0].size();
    vvd tmp(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            tmp[i][j] = h_ReLU(x[i][j]);
        }
    }
    return tmp;
}
vvd hm_tanh(vvd &x) {
    int n = x.size(), m = x[0].size();
    vvd tmp(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            tmp[i][j] = h_tanh(x[i][j]);
        }
    }
    return tmp;
}

vvd hm_softmax(vvd &x) {
    int n = x.size();
    int m = x[0].size();
    vvd y = x;
    for (int i=0; i<n; ++i) {
        double mx = *max_element(y[i].begin(), y[i].end());
        double deno = 0;
        for (int j=0; j<m; ++j) {
            y[i][j] -= mx;
            deno += exp(y[i][j]);
        }
        for (int j=0; j<m; ++j) {
            y[i][j] = exp(y[i][j]) / deno;
        }
    }
    return y;
}
double hm_cross_entropy(vvd &y, vvd &t) {
    int n = y.size(), m = y[0].size();
    double sum = 0;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            if (y[i][j] <= 0) y[i][j] = 1e-5;
            if (t[i][j]) sum += t[i][j] * log(y[i][j]);
        }
    }
    return -sum/n;
}


//  ######     ##       ####   ###  ##
//   ##  ##   ####     ##  ##   ##  ##
//   ##  ##  ##  ##   ##        ## ##
//   #####   ##  ##   ##        ####
//   ##  ##  ######   ##        ## ##
//   ##  ##  ##  ##    ##  ##   ##  ##
//  ######   ##  ##     ####   ###  ##

//  ######   ######    #####   ######     ##       ####     ##     ######    ####     #####   ##   ##
//   ##  ##   ##  ##  ##   ##   ##  ##   ####     ##  ##   ####    # ## #     ##     ##   ##  ###  ##
//   ##  ##   ##  ##  ##   ##   ##  ##  ##  ##   ##       ##  ##     ##       ##     ##   ##  #### ##
//   #####    #####   ##   ##   #####   ##  ##   ##       ##  ##     ##       ##     ##   ##  ## ####
//   ##       ## ##   ##   ##   ##      ######   ##  ###  ######     ##       ##     ##   ##  ##  ###
//   ##       ##  ##  ##   ##   ##      ##  ##    ##  ##  ##  ##     ##       ##     ##   ##  ##   ##
//  ####     #### ##   #####   ####     ##  ##     #####  ##  ##    ####     ####     #####   ##   ##

vvd expansion_bias(vvd &b, int batch) {
    vvd c;
    for (int i=0; i<batch; ++i) {
        c.push_back(b[0]);
    }
    return c;
}

vvd calc_r_cross_entropy(vvd &x, vvd &t) {
    int n = x.size(), m = x[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            for (int k=0; k<m; ++k) {
                if (j == k) tmp[s][j] -= t[s][j] / x[s][j];
                else tmp[s][j] += t[s][k] / (x[s][k]);
            }
            tmp[s][j] /= n;
        }
    }
    return tmp;
}

vvvd calc_r_softmax(vvd &x) {
    int n = x.size(), m = x[0].size();
    vvvd ret(n, vvd(m, vd(m, 0)));
    for (int s=0; s<n; ++s) {
        for (int i=0; i<m; ++i) {
            for (int j=0; j<m; ++j) {
                if (i == j) ret[s][i][j] = x[s][i]*(1 - x[s][j]);
                else ret[s][i][j] = x[s][i]*(0 - x[s][j]);
            }
        }
    }
    return ret;
}

vvd calc_r_ReLU (vvd &a) {
    int n = a.size(), m = a[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            if (a[s][j] >= 0) tmp[s][j] = 1;
        }
    }
    return tmp;
}
vvd calc_r_tanh(vvd &a) {
    int n = a.size(), m = a[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            tmp[s][j] = 4/(exp(-a[s][j]) + exp(a[s][j])) / (exp(-a[s][j]) + exp(a[s][j]));
        }
    }
    return tmp;
}

vvd calc_r_bias (vvd &b, vvd &delta) {
    int n = b.size(), m = b[0].size();
    vvd rb;
    if (n != delta.size() || m != delta[0].size()) cout << "size is not match" << endl;
    rb.assign(1, vd(m, 0));
    for (int j=0; j<m; ++j) {
        for (int i=0; i<n; ++i) {
            rb[0][j] += delta[i][j];
        }
    }
    rb = expansion_bias(rb, n);
    return rb;
}

void updateWeights(vvd &w, vvd &rw, double eta) {
    if (!(w.size() == rw.size() && w[0].size() == rw[0].size())) {
        cout << "The matrix sizes are different." << endl;
        cout << "in update weight" << endl;
        cout << w.size() << ' ' << w[0].size() << endl;
        cout << rw.size() << ' ' << rw[0].size() << endl;
    }
    int n = w.size(), m = w[0].size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            w[i][j] -= eta * rw[i][j];
        }
    }
}
