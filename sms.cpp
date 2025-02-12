#include <iostream>
#include <ctime>
#include <eigen-3.4.0/Eigen/Core>
#include <eigen-3.4.0/Eigen/Dense>
#include <cmath>

using namespace std;
using namespace Eigen;

// int mod q
int mod_q(int a, int q)
{
    int t = a % q;
    if (t < 0)
        return t + q;
    else
        return t;
}

// inv_mod q
int inv_mod(int x, int q)
{
    if (x % q == 0)
        return 0;
    x = x % q;
    while (x < 0)
    {
        x = x + q;
    }
    int y = 1;
    while ((x * y) % q != 1)
    {
        y++;
    }
    return y;
}

// matrix mod q
MatrixXd mod(MatrixXd a, int q)
{
    int n = a.rows();
    int m = a.cols();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            a(i, j) = int(a(i, j)) % q;
        }
    }
    return a;
}

// swap two lines
MatrixXd swap_matrix(MatrixXd a, int i, int j)
{
    int n = a.rows();
    int m = a.cols();
    MatrixXd R = a;
    R.row(i) = a.row(j);
    R.row(j) = a.row(i);
    return R;
}

// upper mod q
MatrixXd upper(MatrixXd mat, int q)
{
    int n = mat.rows();
    int m = mat.cols();
    int time = m < n ? m : n;
    for (int j = 0; j < time; j++)
    {
        int k;
        for (k = j; k < n; k++)
        {
            if (int(mat(k, j)) % q != 0) // 满足这个条件时，认为这个元素不为0
                break;
        }
        if (k < n)
        {
            mat = swap_matrix(mat, j, k); // swap two lines;
            int x = inv_mod(mat(j, j), q);
            // let mat(j,j)=1;
            mat.row(j) = x * mat.row(j);
            for (int v = j; v < m; v++)
            {
                mat(j, v) = mod_q(mat(j, v), q);
                // mat(j, v) = int(mat(j, v)) % q;
            }
            // let mat(u,j)=0 for u>j;
            for (int u = j + 1; u < n; u++)
            {
                mat.row(u) = mat.row(u) - mat(u, j) * mat.row(j);
            }
            // let all 0<=mat(u,v) <q;
            for (int u = j + 1; u < n; u++)
            {
                for (int v = j + 1; v < m; v++)
                {
                    mat(u, v) = mod_q(mat(u, v), q);
                }
            }
        }
    }
    return mat;
}

// 这里U=(A,y) 并且上三角，判断Ax=y是否有解。
bool t_q(MatrixXd U, int q)
{
    int n = U.rows();
    // MatrixXd U = upper(mat, q);
    int s = 0;
    for (int i = 0; i < n; i++)
    {
        if (U(i, i) == 0)
            return 0;
    }
    return 1;
}

// Here U=(A,y) and U is an upper matrix, the output x satisfying Ax=y;
MatrixXd solve_upp(MatrixXd U, int q)
{
    int n = U.rows();
    MatrixXd x = MatrixXd::Zero(n, 1);
    x(n - 1, 0) = U(n - 1, n);
    x(n - 2, 0) = mod_q(U(n - 2, n) - U(n - 2, n - 1) * x(n - 1, 0), q);
    for (int k = n - 1; k >= 0; k--)
    {
        int s = 0;
        for (int i = k + 1; i < n; i++)
        {
            s = s + U(k, i) * x(i, 0);
        }
        x(k, 0) = mod_q(U(k, n) - s, q);
    }
    return x;
}

// verify
bool verify(MatrixXd *P, MatrixXd x, MatrixXd y, int q)
{
    bool ver = 1;
    int m = y.rows();
    for (int k = 0; k < m; k++)
    {
        int y0 = int((x.transpose() * P[k] * x)(0, 0) - y(k, 0));
        if (y0 % q != 0)
        {
            ver = 0;
            break;
        }
    }
    return ver;
}

// generate a random matrix with size (n,m)
MatrixXd generate_random_matrix(int n, int m, int q)
{
    MatrixXd R(n, m);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            R(i, j) = rand() % q;
        }
    }
    return R;
}

// generate an invertible map
MatrixXd invS(int v, int o, int q)
{
    int n = o + v;
    MatrixXd C = MatrixXd::Identity(n, n);
    for (int i = 0; i < v; i++)
    {
        for (int j = v; j < n; j++)
        {
            C(i, j) = rand() % q;
        }
    }
    return C;
}

// secret key generated:F,A,B,S
MatrixXd *Sec(int n, int m, int r, int q)
{
    int o = m + r;
    int v = n - o;
    MatrixXd *myF = new MatrixXd[m + 4];
    for (int k = 0; k < m; k++)
    {
        myF[k] = MatrixXd::Zero(n, n);
        for (int i = 0; i < v; i++)
        {
            for (int j = i; j < n; j++)
            {
                myF[k](i, j) = rand() % q;
                myF[k](j, i) = myF[k](i, j);
            }
        }
    }
    myF[m] = invS(v, o, q);
    myF[m + 1] = generate_random_matrix(n, r, q);
    myF[m + 2] = generate_random_matrix(r, n, q);
    return myF;
}

// public key generated
MatrixXd *Pub(MatrixXd *F, int m, int r, int q)
{

    MatrixXd S = F[m];
    int n = S.rows();
    int o = m + r;
    int v = n - o;
    MatrixXd A = F[m + 1]; // random_matrix(n, r, q);
    MatrixXd B = F[m + 2]; // random_matrix(r, m, q);
    MatrixXd s = S.block(0, v, v, o);

    MatrixXd st = s.transpose();
    MatrixXd St = S.transpose();

    MatrixXd C = St * A * B * S;

    MatrixXd *myP = new MatrixXd[m];
    for (int i = 0; i < m; i++)
    {
        myP[i] = MatrixXd::Zero(n, n);

        MatrixXd F1 = F[i].block(0, 0, v, v);
        MatrixXd F2 = F[i].block(0, v, v, o);

        myP[i].block(0, 0, v, v) = F1;
        myP[i].block(0, v, v, o) = F1 * s + F2;
        myP[i].block(v, 0, o, v) = (myP[i].block(0, v, v, o)).transpose();
        myP[i].block(v, v, o, o) = st * (myP[i].block(0, v, v, o)) + (F2.transpose()) * s;

        myP[i] = C + myP[i];

        myP[i] = mod(myP[i], q);
    }
    return myP;
}

// signature
MatrixXd signature(MatrixXd *F, MatrixXd y, int q)
{
    int m = y.rows();

    MatrixXd A = F[m + 1]; /// random_matrix(n, r, q);
    MatrixXd B = F[m + 2]; // random_matrix(r, n, q);

    int n = A.rows();
    int r = A.cols();
    int o = m + r;
    int v = n - o;

    MatrixXd signature = MatrixXd::Zero(n, 1);
    MatrixXd vinegar = signature.block(0, 0, v, 1);
    MatrixXd U = MatrixXd::Zero(1, r);

    MatrixXd Coeff_y = MatrixXd::Zero(o, o + 1);

    while (t_q(Coeff_y, q) == 0)
    {
        // 随机生成前面v个数
        for (int k = 0; k < v; k++)
        {
            signature(k, 0) = rand() % q;
        }

        // part 1 的随机取值
        for (int k = 0; k < r; k++)
        {
            U(0, k) = rand() % q;
        }

        MatrixXd B0 = U * B;
        vinegar = signature.block(0, 0, v, 1);
        // int b = int((B0 * signature)(0, 0)) % q;
        int b1 = int((B0.block(0, 0, 1, v) * vinegar)(0, 0)) % q;
        //  calculate the coefficients,the first m lines
        for (int k = 0; k < m; k++)
        {
            MatrixXd Fk = vinegar.transpose() * (F[k].block(0, v, v, o));

            int c = int(((vinegar.transpose()) * (F[k].block(0, 0, v, v)) * vinegar)(0, 0));
            Coeff_y(k, o) = y(k, 0) - c - b1;
            for (int j = 0; j < o; j++)
            {
                Coeff_y(k, j) = int(B0(0, j + v) + 2 * Fk(0, j)) % q;
            }
        }
        //  calculate the coefficients,the last r lines
        U = U - vinegar.transpose() * (A.block(0, 0, v, r));
        for (int k = 0; k < r; k++)
        {
            Coeff_y(m + k, o) = U(0, k);
            for (int j = 0; j < o; j++)
            {
                Coeff_y(m + k, j) = A(v + j, k);
            }
        }
        Coeff_y = upper(Coeff_y, q);
    }

    MatrixXd x1 = solve_upp(Coeff_y, q);

    signature.block(v, 0, o, 1) = x1;

    // using the invertible map S
    MatrixXd S = F[m];
    MatrixXd sign = S.inverse() * signature;

    for (int k = 0; k < v; k++)
    {
        sign(k, 0) = int(sign(k, 0)) % q;
    }
    return sign;
}

int main()
{
    int n = 159, m = 40, r = 2, q = 31;
    int times = 1;
    srand(time(0));

    // Generate key time test
    clock_t start_gen_key, end_gen_key;
    start_gen_key = clock();
    for (int t = 0; t < times; t++)
    {
        //  secret key generated,

        MatrixXd *F = Sec(n, m, r, q);
        // MatrixXd A = F[m + 1]; // generate_random_matrix(n, r, q);
        // MatrixXd B = F[m + 2]; // generate_random_matrix(r, n, q);
        // MatrixXd S = F[m];

        // public key generated
        MatrixXd *P = Pub(F, m, r, q);
    }
    end_gen_key = clock();
    double gen_key_time = (end_gen_key - start_gen_key); // CLOCKS_PER_SEC;
    cout << "gen_key_time:" << endl;
    cout << gen_key_time << "um" << endl;
    // cout << gen_key_time << endl;

    // secret key generated
    MatrixXd *F = Sec(n, m, r, q);
    MatrixXd A = F[m + 1]; // generate_random_matrix(n, r, q);
    MatrixXd B = F[m + 2]; // generate_random_matrix(r, n, q);
    MatrixXd S = F[m];

    // public key generated
    MatrixXd *P = Pub(F, m, r, q);

    // Generate signature time test
    clock_t start_gen_signature, end_gen_signature;
    start_gen_signature = clock();
    for (int t = 0; t < times; t++)
    {
        MatrixXd Y(m, 1);
        for (int k = 0; k < m; k++)
        {
            Y(k, 0) = rand() % q;
        }

        MatrixXd sign = signature(F, Y, q);
    }
    end_gen_signature = clock();
    double gen_signature_time = (end_gen_signature - start_gen_signature); // / CLOCKS_PER_SEC;
    cout << "gen_signature_time:" << endl;
    cout << gen_signature_time << "um" << endl;

    // Verification time test

    clock_t start_verify, end_verify;

    MatrixXd Y(m, 1);
    for (int k = 0; k < m; k++)
    {
        Y(k, 0) = rand() % q;
    }
    MatrixXd sign = signature(F, Y, q);

    start_verify = clock();
    for (int t = 0; t < times; t++)
    {

        if (verify(P, sign, Y, q))
        {
            cout << "The signature is valid!" << endl;
        }
        else
        {
            cout << "The signature is not valid!" << endl;
        }
    }
    end_verify = clock();
    double verify_time = (end_verify - start_verify); // / CLOCKS_PER_SEC;
    cout << "verify_time:" << endl;
    cout << verify_time << "mu" << endl;

    // The correctness test
    for (int t = 0; t < times; t++)
    {
        MatrixXd Y(m, 1);
        for (int k = 0; k < m; k++)
        {
            Y(k, 0) = rand() % q;
        }
        MatrixXd sign = signature(F, Y, q);

        if (verify(P, sign, Y, q))
        {
            cout << "The method is successful!" << endl;
        }
        else
        {
            cout << "The method is failed!" << endl;
        }
    }
    
    return 0;
}