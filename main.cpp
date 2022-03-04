#include <cmath>
#include <iostream>
#include <vector>
#include "mkl.h"
#include "mkl_lapacke.h"

using namespace std;

void manifold_projection(vector<double>& normal_vec, vector<double>& extern_force, vector<double>& tangent_vec)
{
    if(normal_vec.size()!=extern_force.size() || normal_vec.size()!=tangent_vec.size() || extern_force.size()!=tangent_vec.size())
    {
        cout << "normal_vec = " << normal_vec.size() << " extern_force = " << extern_force.size() << " tangent_vec = " << tangent_vec.size() << "\n";
    }

    int n = normal_vec.size();

    double norm = 0.0;
    for(int i=0;i<n;i++)
    {
        norm += pow(normal_vec[i], 2);
    }
    norm = pow(norm, 0.5);

    vector<double> normed_normal_vec(n);
    for(int i=0;i<n;i++)
    {
        normed_normal_vec[i] = normal_vec[i] / norm;
    }

    vector<double> projection(n, 0);

    double a = cblas_ddot(n, &*normed_normal_vec.begin(), 1, &*extern_force.begin(), 1);
    cblas_daxpy(n, a, &*normed_normal_vec.begin(), 1, &*projection.begin(), 1);
    for(int i=0;i<n;i++)
    {
        tangent_vec[i] = extern_force[i] - projection[i];
    }

}

int main(void)
{
	vector<double> normal_vec{1.0, 1.0, 1.0};
	vector<double> extern_force{2.0, 1.0, 2.0};
	vector<double> result(3);

	manifold_projection(normal_vec, extern_force, result);

	for (int i = 0; i < 3; i++)
	{
		cout << result[i] << ' ';
	}
	cout << endl;
	
}