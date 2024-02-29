/*
入力(p, q)が
中心原点，半径３の円の
内側にあるか外側にあるかを判定する
ニューラルネットワーク
*/
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <set>
#include <math.h>
#include <string>
#include <fstream>
#include <random>

using namespace std;

#define pi 3.14159265358979323846
#define yes "Yes"
#define no "No"
#define yesno(bool) if(bool){cout<<"Yes"<<endl;}else{cout<<"No"<<endl;}
#define tp() cout << "here~~" << endl

using vd = vector<double>;
using vvd = vector<vector<double>>;

random_device rd;
long long seed = 0;//rd()
// long long seed = rd();
mt19937 engine(seed);

double gaussianDistribution (double mu, double sig) {
    normal_distribution <> dist(mu, sig);
    return dist(engine);
}
//rd()は実行毎に異なる
//rand()は実行毎に同じ
mt19937 gen(seed);//seed
uniform_real_distribution<> distCircle(-6, 6);

double h_sigmoid(double x) {
    return 1/(1+exp(-x));
}
double h_tash(double x) {
    return (exp(x)-exp(-x)) / (exp(x)+exp(-x));
}
double h_ReLU(double x) {
    return (x > 0) ? x : 0;
}
vvd h_ReLUMatrix(vvd &x) {
    int n = x.size(), m = x[0].size();
    vvd tmp(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            tmp[i][j] = h_ReLU(x[i][j]);
        }
    }
    return tmp;
}
void showMatrix(vvd &a) {
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

void showMatrixB(vvd &a) {
    int m = a[0].size();
    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;

    for (int j=0; j<m; ++j) cout << a[0][j] << ' ';
    cout << endl;

    for (int j=0; j<m; ++j) cout << "--";
    cout << endl;
}

vvd softMax(vvd &x) {
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
double crossEntropy(vvd &y, vvd &t) {
    int n = y.size(), m = y[0].size();
    double sum = 0;
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            if (y[i][j] == 0) {
                cout << "log 0!!!!!!!!!!!!!!!!" << endl;
            } else if (y[i][j] < 0) {
                cout << "log -x !!!!!!!!!!!!!!!!" << endl;
            }
            // if (y[i][j] <= 0) y[i][j] = 1e-5;
            if (t[i][j]) sum += t[i][j] * log(y[i][j]);
        }
    }
    if (sum != sum) cout << "sum is nanananananananan" << endl;
    return -sum/n;
}
// c = a * b
void multiMatrix(vvd &c, vvd &a, vvd &b) {
    int n = a.size(), m = b.size(), l = b[0].size();
    c.assign(n, vd(l, 0));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<l; ++j) {
            for (int k=0; k<m; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}
// c = a .* b admal
void admMultiMatrix(vvd &c, vvd &a, vvd &b) {
    int n = a.size(), m = a[0].size();
    c.assign(n, vd(m));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            c[i][j] = a[i][j] * b[i][j];
        }
    }
}
// c = a + b
void addMatrix(vvd& c, vvd &a, vvd &b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        cout << "The matrix sizes are different." << endl;
        return;
    }
    int n = a.size(), m = a[0].size();
    c.assign(n, vd(m, 0));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            c[i][j] += a[i][j] + b[i][j];
        }
    }
}
void tMatrix(vvd &a) {
    int n = a.size(), m = a[0].size();
    vvd t = a;
    a.assign(m, vd(n));
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            a[j][i] = t[i][j];
        }
    }
}
// bool judgeTerm(double x, double y){ return (x*x + y*y < 9) ? true : false;}
bool judgeTerm(double x, double y){ return (y > x) ? true : false;}
void makeData(vvd &x, int n, int seed=0) {
    //条件を満たす点と満たさない点をｎ個ずつ作る
    
    x.assign(2*n, vd(2, 0));
    int id = 0;
    while(id < n) {
        double a, b;
        a = distCircle(gen);
        b = distCircle(gen);
        if (judgeTerm(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
    while(id < n*2) {
        double a, b;
        a = distCircle(gen);
        b = distCircle(gen);
        if (!judgeTerm(a, b)) {
            x[id][0] = a;
            x[id][1] = b;
            ++id;
        }
    }
}
void makeInitialValue(vvd &table, double mu, double sig) {
    int n = table.size(), m = table[0].size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            table[i][j] = gaussianDistribution(mu, sig);
        }
    }
}


void expansionBias(vvd &b, int batch) {
    vd tmp = b[0];
    if (b.size() != 1) {
        cout << "bias batch size error" << endl;
        return ;
    }
    for (int i=0; i<batch-1; ++i) {
        b.push_back(tmp);
    }
}

vvd calc_r_hL_x3(vvd &x, vvd &t) {
    int n = x.size(), m = x[0].size();
    cout << "n " << n << endl;
    // double ips = 0.01, fLup, fLdown;
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {//1
        for (int j=0; j<m; ++j) {//2
            // tmp[s][j] = -t[s][j] / x[s][j];
            for (int k=0; k<m; ++k) {
                if (j == k) tmp[s][j] -= t[s][j] / x[s][j];
                else tmp[s][j] += t[s][k] / (x[s][k]);
            }
            tmp[s][j] /= n;
        }
    }
    return tmp;
}
vvd calc_r_h3_a3 (vvd &a, vvd &x) {
    int n = a.size(), m = a[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            tmp[s][j] = x[s][j]*(1-x[s][j]);
        }
    }
    return tmp;
}

vvd calc_r_h2_a2 (vvd &a, vvd &x) {
    int n = a.size(), m = a[0].size();
    vvd tmp(n, vd(m, 0));
    for (int s=0; s<n; ++s) {
        for (int j=0; j<m; ++j) {
            if (a[s][j] >= 0) tmp[s][j] = 1;
        }
    }
    return tmp;
}

void calc_r_L_b (vvd &rb, vvd &b, vvd &delta) {
    int n = b.size(), m = b[0].size();
    if (n != delta.size() || m != delta[0].size()) {
        cout << "size is not match" << endl;
    }
    rb.assign(1, vd(m, 0));
    for (int j=0; j<m; ++j) {
        for (int i=0; i<n; ++i) {
            rb[0][j] += delta[i][j];
        }
    }
    expansionBias(rb, n);
}

void updateWeights(vvd &w, vvd &rw, double eta) {
    if (!(w.size() == rw.size() && w[0].size() == rw[0].size())) {
        cout << "the sizes are different" << endl;
    }
    int n = w.size(), m = w[0].size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            w[i][j] -= eta * rw[i][j];
        }
    }
}

double calcAccuracyRate(vvd &y, vvd &t) {
    int n = y.size();
    double sum = 0;
    for (int i=0; i<n; ++i) {
        if (y[i][0] < y[i][1] && t[i][1] || y[i][0] > y[i][1] && t[i][0]) sum += 1;
    }
    return sum / n;
}

void shuffleVVD(vvd &v, vector<int> &id) {
    vvd tmp = v;
    int n = v.size();
    for (int i=0; i<n; ++i) {
        for (int j=0; j<v[0].size(); ++j) {
            tmp[i][j] = v[id[i]][j];
        }
    }
    v = tmp;
}
#include <fstream>
void outputfile(vvd &x) {
    int n = x.size(), m = x[0].size();
    string fname1 = "in.txt";
    string fname2 = "out.txt";
    ofstream outputFile (fname1);
    ofstream outputFile2 (fname2);
    

    for (int i=0; i<n; ++i) {
        for (int j=0; j<m; ++j) {
            if (i < n/2) {
                outputFile << x[i][j];
                if (j != m-1) outputFile << " ";
            } else {
                outputFile2 << x[i][j];
                if (j != m-1) outputFile2 << " ";
            }
            
        }
        if (i < n/2) {
            outputFile << endl;
        } else {
            outputFile2 << endl;
        }
        
    } 
}


int main() {
    vvd x0, x1, x2, x3, a1, a2, a3, w1, w2, w3, b1, b2, b3;
    vvd tmp1, r_hL_x3, r_h3_a3, Delta3, r_L_w3, tx2, r_h2_a2, tw3, tmp2, Delta2, tx1, r_L_w2, r_h1_a1, tw2, tmp3, Delta1, tx0, r_L_w1, r_L_b1, r_L_b2, r_L_b3;
    double eta = 0.1;
    int n = 1;

    vector<int> id(n);
    for (int i=0; i<n; ++i) id[i] = i;
    shuffle(id.begin(), id.end(), engine);
    
    w1 = {{1, -2, 3}, {-1, 2, -3}};
    b1 = {{1, -2, 3}};
    w2 = {{-1, 2, -3}, {1, -2, 3}, {-1, 2, -3}};
    b2 = {{1, -2, 3}};
    w3 = {{-1, 2}, {-3, 1}, {-2, 3}};
    b3 = {{34, -54}};

    cout << "w" << endl;
    showMatrix(w1);
    showMatrix(w2);
    showMatrix(w3);

    cout << "b" << endl;
    showMatrix(b1);
    showMatrix(b2);
    showMatrix(b3);
    
    // return 0;

    //教師データの作成 {1,0}内側
    vvd t = {{1, 0}};
    x0 = {{-1, 2}};
    

    cout << "first x0" << endl;
    showMatrix(x0);
    cout << "t" << endl;
    showMatrix(t);


    //learn
    for (int i=0; i<2   ; ++i) {
        //forward propagation--------------------------
        
        //a1 = w1 * x0 + b1
        multiMatrix(tmp1, x0, w1);
        addMatrix(a1, tmp1, b1);
        cout << "A1" << endl;
        showMatrix(a1);

        x1 = h_ReLUMatrix(a1);
        cout << "X1" << endl;
        showMatrix(x1);

        //a2 = w2 * x1 + b2
        multiMatrix(tmp1, x1, w2);
        addMatrix(a2, tmp1, b2);
        cout << "A2" << endl;
        showMatrix(a2);

        x2 = h_ReLUMatrix(a2);
        cout << "X2" << endl;
        showMatrix(x2);

        //a3 = w3 * x2 + b3
        multiMatrix(tmp1, x2, w3);
        addMatrix(a3, tmp1, b3);
        cout << "A3" << endl;
        showMatrix(a3);

        x3 = softMax(a3);
        cout << "X3" << endl;
        showMatrix(x3);

        
        cout << "cross entropy";
        cout << crossEntropy(x3, t) << endl;

        //back propagation--------------------------------
        r_hL_x3 = calc_r_hL_x3(x3, t);
        cout << "r_fL_x3" << endl;
        showMatrix(r_hL_x3);

        r_h3_a3 = calc_r_h3_a3(a3, x3);
        cout << "r_h3_a3" << endl;
        showMatrix(r_h3_a3);

        admMultiMatrix(Delta3, r_hL_x3, r_h3_a3);
        cout << "Delta3" << endl;
        showMatrix(Delta3);

        calc_r_L_b(r_L_b3, b3, Delta3);
        cout << "r_L_b3" << endl;
        showMatrix(r_L_b3);

        tx2 = x2;
        tMatrix(tx2);
        multiMatrix(r_L_w3, tx2, Delta3);
        cout << "r_L_w3" << endl;
        showMatrix(r_L_w3);


        r_h2_a2 = calc_r_h2_a2(a2, x2);
        cout << "r_h2_a2" << endl;
        showMatrix(r_h2_a2);

        tw3 = w3;
        tMatrix(tw3);
        multiMatrix(tmp2, Delta3, tw3);
        admMultiMatrix(Delta2, r_h2_a2, tmp2);
        cout << "Delta2" << endl;
        showMatrix(Delta2);


        calc_r_L_b(r_L_b2, b2, Delta2);
        cout << "r_L_b2" << endl;
        showMatrix(r_L_b2);

        tx1 = x1;
        tMatrix(tx1);
        multiMatrix(r_L_w2, tx1, Delta2);
        cout << "r_L_w2" << endl;
        showMatrix(r_L_w2);


        r_h1_a1 = calc_r_h2_a2(a1, x1);
        cout << "r_h1_a1" << endl;
        showMatrix(r_h1_a1);


        tw2 = w2;
        tMatrix(tw2);
        multiMatrix(tmp3, Delta2, tw2);
        admMultiMatrix(Delta1, r_h1_a1, tmp3);
        cout << "Delta1" << endl;
        showMatrix(Delta1);


        calc_r_L_b(r_L_b1, b1, Delta1);
        cout << "r_L_b1" << endl;
        showMatrix(r_L_b1);

        tx0 = x0;
        tMatrix(tx0);
        multiMatrix(r_L_w1, tx0, Delta1);
        cout << "r_L_w1" << endl;
        showMatrix(r_L_w1);

        updateWeights(w1, r_L_w1, eta);
        updateWeights(w2, r_L_w2, eta);
        updateWeights(w3, r_L_w3, eta);
        updateWeights(b1, r_L_b1, eta);
        updateWeights(b2, r_L_b2, eta);
        updateWeights(b3, r_L_b3, eta);

        cout << "w" << endl;
        showMatrix(w1);
        showMatrix(w2);
        showMatrix(w3);

        cout << "b" << endl;
        showMatrix(b1);
        showMatrix(b2);
        showMatrix(b3);
    }

    // test set
    cout << "test" << endl;
    expansionBias(b1, n);
    expansionBias(b2, n);
    expansionBias(b3, n);
    makeData(x0, n/2);
    
    

    cout << "test x0" << endl;
    // showMatrix(x0);

    //a1 = w1 * x0 + b1
    multiMatrix(tmp1, x0, w1);
    addMatrix(a1, tmp1, b1);
    x1 = h_ReLUMatrix(a1);
    //a2 = w2 * x1 + b2
    multiMatrix(tmp1, x1, w2);
    addMatrix(a2, tmp1, b2);
    x2 = h_ReLUMatrix(a2);
    //a3 = w3 * x2 + b3
    multiMatrix(tmp1, x2, w3);
    addMatrix(a3, tmp1, b3);
    x3 = softMax(a3);

    cout << "cross entropy";
    cout << crossEntropy(x3, t) << endl;
    cout << "accuracy rate ";
    cout << calcAccuracyRate(x3, t) << endl;

    showMatrix(w1);
    showMatrix(w2);
    showMatrix(w3);
    showMatrixB(b1);
    showMatrixB(b2);
    showMatrixB(b3);
    
    
    

}


/*


*/