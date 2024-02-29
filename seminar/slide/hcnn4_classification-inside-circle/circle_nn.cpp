/*
入力(p, q)が
中心原点，半径３の円の
内側にあるか外側にあるかを判定する
ニューラルネットワーク
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

//function
bool judge_term(double x, double y);
vvd make_data(int n);
void make_initial_value(vvd &table, double mu, double sig);
double calc_accuracy_rate(vvd &y, vvd &t);
void shuffle_VVD(vvd &v, vector<int> &id);
void show_parameters(vector<layer_t> &nn, int depth);
void outputfile(const vvd &output);
void drawing_by_python(vector<layer_t> &nn, int depth);
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
vvd hm_tanh(vvd &x);
vvd hm_softmax(vvd &x);
double hm_cross_entropy(vvd &y, vvd &t);
//BACK PROPAGATION
vvd expansion_bias(vvd &b, int batch);
vvd calc_r_cross_entropy(vvd &x, vvd &t);
vvvd calc_r_softmax(vvd &x);
vvd calc_r_ReLU (vvd &a);
vvd calc_r_tanh(vvd &a);
vvd calc_r_bias (vvd &delta);
void updateWeights(vvd &w, vvd &rw, double eta);

random_device rd;
long long SEED = 0;//実行毎に同じ乱数生成
// long long SEED = rd();//実行毎に異なる乱数生成
mt19937 engine(SEED);
uniform_real_distribution<> distCircle(-6, 6);

//  ##   ##    ##      ####    ##   ##
//  ### ###   ####      ##     ###  ##
//  #######  ##  ##     ##     #### ##
//  #######  ##  ##     ##     ## ####
//  ## # ##  ######     ##     ##  ###
//  ##   ##  ##  ##     ##     ##   ##
//  ##   ##  ##  ##    ####    ##   ##

int main() {
    vvd x, t;
    double eta = 0.03, attenuation = 0.6;
    int n = 1000;
    int show_interval = 1000;
    int learning_plan = 2000;
    int loop = 9500;//9500
    int batch_size = 100;
    vector<int> nn_form = {2, 3, 3, 2};
    int depth = nn_form.size()-1;

    vector<layer_t> nn(depth);

    vector<int> id(n);
    for (int i=0; i<n; ++i) id[i] = i;
    
    //Heの初期化
    for (int i=0; i<depth; ++i) {
        nn[i].w.assign(nn_form[i], vd(nn_form[i+1], 0));
        nn[i].b.assign(batch_size, vd(nn_form[i+1], 0));
        make_initial_value(nn[i].w, 0, sqrt(2.0/nn_form[i]));
        make_initial_value(nn[i].b, 0, sqrt(2.0/nn_form[i]));
        nn[i].b = expansion_bias(nn[i].b, batch_size);
    }
    
    //初期のパラメータの表示
    cout << "first parameters" << endl;
    // show_parameters(nn, depth);
    
    //訓練セットの作成
    //前半半分は円の内側 後半半分は円の外側
    x = make_data(n);
    //教師ラベルの作成 {1,0}内側 {0,1}外側
    for (int i=0; i<n/2; ++i) t.push_back({1, 0});//inside
    for (int i=0; i<n/2; ++i) t.push_back({0, 1});//outside
    
    //learn
    for (int i=0; i<loop; ++i) {
        //mini batchの作成
        vvd x0, t0;
        shuffle(id.begin(), id.end(), engine);
        shuffle_VVD(t, id);
        shuffle_VVD(x, id);
        //全データから先頭batchi_size個だけmini batchを取得
        for (int j=0; j<batch_size; ++j) {
            x0.push_back(x[j]);
            t0.push_back(t[j]);
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
                // vvd r_h_a = calc_r_ReLU(nn[k].a);
                vvd r_h_a = calc_r_tanh(nn[k].a);
                nn[k].delta = matrix_adm_multi(r_h_a, matrix_multi(nn[k+1].delta, matrix_t(nn[k+1].w)));
            }
            nn[k].rb = calc_r_bias(nn[k].delta);
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
        nn[i].b = expansion_bias(nn[i].b, n);
    }
    //forward propagation
    for (int k=0; k<depth; ++k) {
        if (k == 0) nn[k].a = matrix_add(matrix_multi(x, nn[k].w), nn[k].b);
        else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
        // if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
        if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
        else nn[k].x = hm_softmax(nn[k].a);
    }
    cout << " cross entropy ";
    cout << hm_cross_entropy(nn[depth-1].x, t) << endl;
    cout << "accuracy rate ";
    cout << calc_accuracy_rate(nn[depth-1].x, t) << endl;
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;

    // test set-------------------------------------
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;
    cout << "test set" << endl;
    //新しいデータをランダムに作成
    x = make_data(n);
    //教師ラベルも作成
    t.assign(0, vd(0));
    for (int i=0; i<n/2; ++i) t.push_back({1, 0});
    for (int i=0; i<n/2; ++i) t.push_back({0, 1});
    
    //test set は単に順伝播させて，正解率を見るだけだからシャッフルは必要ない

    //forward propagation
    for (int k=0; k<depth; ++k) {
        if (k == 0) nn[k].a = matrix_add(matrix_multi(x, nn[k].w), nn[k].b);
        else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
        // if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
        if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
        else nn[k].x = hm_softmax(nn[k].a);
    }
    cout << " cross entropy ";
    cout << hm_cross_entropy(nn[depth-1].x, t) << endl;
    cout << "accuracy rate ";
    cout << calc_accuracy_rate(nn[depth-1].x, t) << endl;
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;

    //最後のパラメータの表示
    cout << "last parameters" << endl;
    show_parameters(nn, depth);

    //一旦csvに出力したのちpythonで描画してみる
    drawing_by_python(nn, depth);
}

//  #######  ##   ##  ##   ##    ####   ######    ####     #####   ##   ##
//   ##   #  ##   ##  ###  ##   ##  ##  # ## #     ##     ##   ##  ###  ##
//   ## #    ##   ##  #### ##  ##         ##       ##     ##   ##  #### ##
//   ####    ##   ##  ## ####  ##         ##       ##     ##   ##  ## ####
//   ## #    ##   ##  ##  ###  ##         ##       ##     ##   ##  ##  ###
//   ##      ##   ##  ##   ##   ##  ##    ##       ##     ##   ##  ##   ##
//  ####      #####   ##   ##    ####    ####     ####     #####   ##   ##

// bool judge_term(double x, double y){ return (x*x + y*y < 9) ? true : false;}
// bool judge_term(double x, double y) { return (x * y > 0 ? true : false);}//xor
// bool judge_term(double x, double y) { return (y > -x) ? true : false;}//linear
bool judge_term(double x, double y) { return (y > x*x ? true : false);}//sqare
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

void show_parameters(vector<layer_t> &nn, int depth) {
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;
    for (int i=0; i<depth; ++i) {
        cout << "w " << i+1 << endl; 
        matrix_show(nn[i].w);
    }
    for (int i=0; i<depth; ++i) {
        cout << "b " << i+1 << endl;
        matrix_show_b(nn[i].b);
    }
    for (int i=0; i<40; ++i) cout << "=";
    cout << endl;
}
//x, y, tを列挙
void outputfile(const vvd &output) {
    int n = output.size(), m = output[0].size();
    string fname = "circle_.csv";
    ofstream outputFile (fname);

    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            outputFile << output[i][j];
            if (j != m-1) outputFile << ", ";
        }
        outputFile << endl;
    } 
}
void drawing_by_python(vector<layer_t> &nn, int depth) {
    vvd data;
    for (int i=0; i<depth; ++i) {
        nn[i].b = expansion_bias(nn[i].b, 1);
    }
    for (double x=-6; x<=6; x+=0.05) {
        for (double y=-6; y<=6; y+=0.05) {
            vvd tmp = {{x, y}};
            //forward propagation
            for (int k=0; k<depth; ++k) {
                if (k == 0) nn[k].a = matrix_add(matrix_multi(tmp, nn[k].w), nn[k].b);
                else nn[k].a = matrix_add(matrix_multi(nn[k-1].x, nn[k].w), nn[k].b);
                // if (k < depth-1) nn[k].x = hm_ReLU(nn[k].a);
                if (k < depth-1) nn[k].x = hm_tanh(nn[k].a);
                else nn[k].x = hm_softmax(nn[k].a);
            }
            if (nn[depth-1].x[0][0] > nn[depth-1].x[0][1]) {
                //inside
                tmp[0].push_back(1);
            } else {
                //outside
                tmp[0].push_back(0);
            }
            data.push_back(tmp[0]);
        }
    }
    outputfile(data);
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
                else tmp[s][j] += t[s][k] / x[s][k];
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

vvd calc_r_bias (vvd &delta) {
    int n = delta.size(), m = delta[0].size();
    vvd rb;
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
