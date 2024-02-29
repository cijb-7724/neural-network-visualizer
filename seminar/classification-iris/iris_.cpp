/*
iris datasets
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

//function
bool judge_term(double x, double y);
vvd make_data(int n);
void make_initial_value(vvd &table, double mu, double sig);
double calc_accuracy_rate(vvd &y, vvd &t);
void shuffle_VVD(vvd &v, vector<int> &id);
vvd readCSV(string filename);
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
double h_tash(double x);
double h_ReLU(double x);
vvd hm_ReLU(vvd &x);
vvd hm_softmax(vvd &x);
double hm_cross_entropy(vvd &y, vvd &t);
//BACK PROPAGATION
vvd expansion_bias(vvd &b, int batch);
vvd calc_r_cross_entropy(vvd &x, vvd &t);
vvvd calc_r_softmax(vvd &x);
vvd calc_r_ReLU (vvd &a);
vvd calc_r_bias (vvd &b, vvd &delta);
void updateWeights(vvd &w, vvd &rw, double eta);

typedef struct {
    vvd w;
    vvd b;
    vvd a;
    vvd x;
    vvd delta;
    vvd rw;
    vvd rb;
} layer_t;

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
all 150
train 105:35, 35, 35
test 45:15, 15, 15

train 93:31, 31, 31
valid 12:4, 4, 4
test 45:15, 15, 15
50, 50, 50
*/

vvd normalization(const vvd & data) {
    int n = data.size();
    int m = data[0].size();
    vvd ret(n, vd(m));
    for (int j=0; j<m; ++j) {
        vd tmp;
        for (int i=0; i<n; ++i) {
            // cout << data[j][i] << ' ';
            tmp.push_back(data[i][j]);
        }
        double mn = *min_element(tmp.begin(), tmp.end());
        double mx = *max_element(tmp.begin(), tmp.end());
        for (int i=0; i<n; ++i) {
            ret[i][j] = (data[i][j] - ((mx+mn)/2)) / ((mx - mn)/2);
        }
    }
    return ret;
}

int main() {
    //data import
    ////////////////////////////////////////////////////////////////
    vvd data = readCSV("data/iris_datasets_data.csv");
    cout << data.size() << ' ' << data[0].size() << endl;
    vvd target_tmp = readCSV("data/iris_datasets_target.csv");
    vvd target;
    for (int i=0; i<target_tmp[0].size(); ++i) {
        if (target_tmp[0][i] == 0) target.push_back({1, 0, 0});
        else if (target_tmp[0][i] == 1) target.push_back({0, 1, 0});
        else if (target_tmp[0][i] == 2) target.push_back({0, 0, 1});
    }
    cout << target.size() << ' ' << target[0].size() << endl;
    ////////////////////////////////////////////////////////////////

    data = normalization(data);

    int train_size = 105, test_size = 45;
    int all_size = train_size + test_size;
    vvd train_x, train_t, test_x, test_t;
    
    vector<int> id(all_size/3);
    for (int i=0; i<all_size/3; ++i) id[i] = i;

    //data分布を保持しつつ，ランダムにtrain setとtest setに分類する
    for (int i=0; i<3; ++i) {
        vvd tmpX, tmpT;
        for (int j=0; j<all_size/3; ++j) {
            tmpX.push_back(data[i*(all_size/3)+j]);
            tmpT.push_back(target[i*(all_size/3)+j]);
        }
        shuffle(id.begin(), id.end(), engine);
        shuffle_VVD(tmpX, id);
        shuffle_VVD(tmpT, id);
        for (int j=0; j<all_size/3; ++j) {
            if (j < train_size/3) {
                train_x.push_back(tmpX[j]);
                train_t.push_back(tmpT[j]);
            } else {
                test_x.push_back(tmpX[j]);
                test_t.push_back(tmpT[j]);
            }
        }
    }
    /*
    このタイミングでは各クラス内ではシャッフルされているが，
    クラスは昇順に並んでいる．
    learnの中，各ステップでシャッフルする．
    */
    

    
    
    double eta = 3, attenuation = 0.9;
    int show_interval = 100;
    int learning_plan = 100;
    int loop = 1000;
    int batch_size = 100; //<train_size
    vector<int> nn_form = {4, 10, 10, 3};
    // vector<int> nn_form = {4, 100, 100, 100, 100, 100, 3};
    int depth = nn_form.size()-1;

    vector<layer_t> nn(depth);

    id.assign(train_size, 0);
    for (int i=0; i<train_size; ++i) id[i] = i;
    
    //Heの初期化
    for (int i=0; i<depth; ++i) {
        nn[i].w.assign(nn_form[i], vd(nn_form[i+1], 0));
        nn[i].b.assign(1, vd(nn_form[i+1], 0));
        make_initial_value(nn[i].w, 0, sqrt(2.0/nn_form[i]));
        make_initial_value(nn[i].b, 0, sqrt(2.0/nn_form[i]));
        nn[i].b = expansion_bias(nn[i].b, batch_size);
    }
    
    //初期のパラメータの表示
    cout << "first parameters" << endl;
    // for (int i=0; i<40; ++i) cout << "=";
    // cout << endl;
    // for (int i=0; i<depth; ++i) {
    //     cout << "w " << i+1 << endl; 
    //     matrix_show(nn[i].w);
    // }
    // for (int i=0; i<depth; ++i) {
    //     cout << "b " << i+1 << endl;
    //     matrix_show_b(nn[i].b);
    // }
    // for (int i=0; i<40; ++i) cout << "=";
    // cout << endl;
    
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
  

        // cout << "=========================================" << endl;
        // cout << "i = " << i << " before forward propagation parameters" << endl;
        // cout << "w" << endl;
        // for (int i=0; i<depth; ++i) matrix_show(nn[i].w);
        // cout << "b" << endl;
        // for (int i=0; i<depth; ++i) matrix_show_b(nn[i].b);
        // cout << "=========================================" << endl;
        

        //forward propagation
        for (int k=0; k<depth; ++k) {
            if (k == 0) nn[k].a = matrix_add(matrix_multi(x0, nn[k].w), nn[k].b);
            else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
            if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
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
                r_h_a = calc_r_ReLU(nn[k].a);
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
        if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
        else nn[k].x = hm_softmax(nn[k].a);
    }
    cout << " cross entropy ";
    cout << hm_cross_entropy(nn[depth-1].x, train_t) << endl;
    cout << "accuracy rate ";
    cout << calc_accuracy_rate(nn[depth-1].x, train_t) << endl;
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;

    // test set-------------------------------------
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;
    cout << "test set" << endl;
    for (int i=0; i<depth; ++i) {
        nn[i].b = expansion_bias(nn[i].b, test_size);
    }
    
    //test set は単に順伝播させて，正解率を見るだけだからシャッフルは必要ない

    //forward propagation
    for (int k=0; k<depth; ++k) {
        if (k == 0) nn[k].a = matrix_add(matrix_multi(test_x, nn[k].w), nn[k].b);
        else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
        if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
        else nn[k].x = hm_softmax(nn[k].a);
    }
    cout << " cross entropy ";
    cout << hm_cross_entropy(nn[depth-1].x, test_t) << endl;
    cout << "accuracy rate ";
    cout << calc_accuracy_rate(nn[depth-1].x, test_t) << endl;
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;

    //最後のパラメータの表示
    // cout << "last parameters" << endl;
    // for (int i=0; i<40; ++i) cout << "=";
    // cout << endl;
    // for (int i=0; i<depth; ++i) {
    //     cout << "w " << i+1 << endl; 
    //     matrix_show(nn[i].w);
    // }
    // for (int i=0; i<depth; ++i) {
    //     cout << "b " << i+1 << endl;
    //     matrix_show_b(nn[i].b);
    // }
    // for (int i=0; i<40; ++i) cout << "=";
    // cout << endl;
}


//  #######  ##   ##  ##   ##    ####   ######    ####     #####   ##   ##
//   ##   #  ##   ##  ###  ##   ##  ##  # ## #     ##     ##   ##  ###  ##
//   ## #    ##   ##  #### ##  ##         ##       ##     ##   ##  #### ##
//   ####    ##   ##  ## ####  ##         ##       ##     ##   ##  ## ####
//   ## #    ##   ##  ##  ###  ##         ##       ##     ##   ##  ##  ###
//   ##      ##   ##  ##   ##   ##  ##    ##       ##     ##   ##  ##   ##
//  ####      #####   ##   ##    ####    ####     ####     #####   ##   ##

bool judge_term(double x, double y){ return (x*x + y*y < 9) ? true : false;}
//条件を満たす点と満たさない点をn/2個ずつ作る
vvd make_data(int n) {
    vvd x;
    x.assign(n, vd(2, 0));
    int id = 0;
    while(id < n/2) {
        double a, b;
        a = distCircle(engine);
        b = distCircle(engine);
        if (judge_term(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
    while(id < n) {
        double a, b;
        a = distCircle(engine);
        b = distCircle(engine);
        if (!judge_term(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
    return x;
}
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
vvd readCSV(string filename) {
    ifstream file(filename);
    vvd data;
    if (!file.is_open()) {
        cerr << "Error Opening File!" << endl;
        return data;
    }

    string line;
    while (getline(file, line)) {
        vd row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    file.close();
    return data;
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
double h_tash(double x) {
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
            // if (y[i][j] <= 0) y[i][j] = 1e-5;
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
//rx_k/ra_j
//m class 分類
//m次正方行列を返す
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
