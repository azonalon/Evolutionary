#include <iostream>
#include <vector>
#include <array>
#include "physics/ElasticModel.hpp"
#include <Eigen/Dense>
using namespace std;
std::vector<double> vertices =
{1., 0., 0.973045, -0.230616, 0.893633, -0.448799,
  0.766044, -0.642788, 0.597159, -0.802123, 0.39608, -0.918216,
  0.173648, -0.984808, -0.0581448, -0.998308, -0.286803, -0.95799,
-0.5, -0.866025, -0.686242, -0.727374, -0.835488, -0.549509,
-0.939693, -0.34202, -0.993238, -0.116093, -0.993238,
  0.116093, -0.939693, 0.34202, -0.835488, 0.549509, -0.686242,
  0.727374, -0.5, 0.866025, -0.286803, 0.95799, -0.0581448, 0.998308,
  0.173648, 0.984808, 0.39608, 0.918216, 0.597159, 0.802123, 0.766044,
   0.642788, 0.893633, 0.448799, 0.973045, 0.230616, 4., 4., 4., 3.,
  3., 3., 3., 4., 3., 2., 2.97304, 1.76938, 2.89363, 1.5512, 2.76604,
  1.35721, 2.59716, 1.19788, 2.39608, 1.08178, 2.17365, 1.01519,
  1.94186, 1.00169, 1.7132, 1.04201, 1.5, 1.13397, 1.31376, 1.27263,
  1.16451, 1.45049, 1.06031, 1.65798, 1.00676, 1.88391, 1.00676,
  2.11609, 1.06031, 2.34202, 1.16451, 2.54951, 1.31376, 2.72737, 1.5,
  2.86603, 1.7132, 2.95799, 1.94186, 2.99831, 2.17365, 2.98481,
  2.39608, 2.91822, 2.59716, 2.80212, 2.76604, 2.64279, 2.89363,
  2.4488, 2.97304, 2.23062, -0.0403894, -0.22906, 1.76741,
  2., -0.4947, -0.248448, 0.290132, -0.502523, 0.443185,
  0.0518009, -0.157739, 0.365681, 1.55549, 1.62701, 2.12797, 1.57254,
  2.33273, 2.21884, 1.79212, 2.48193, 0.278453, 0.482295, 2.52332,
  1.80953};
std::vector<std::array<unsigned, 3>> indices =
   {{8, 7, 58}, {9, 8, 58}, {60, 10, 9}, {58, 60, 9}, {11,
   60, 12}, {6, 61, 7}, {61, 5, 4}, {10, 60, 11}, {1, 0, 62}, {62, 61,
    2}, {68, 62, 25}, {1, 62, 2}, {6, 5, 61}, {26, 62, 0}, {15, 14,
   60}, {60, 14, 13}, {16, 15, 63}, {59, 66, 67}, {17, 63, 18}, {18,
   63, 19}, {63, 15, 60}, {31, 69, 32}, {22, 21, 68}, {60, 58,
   63}, {68, 24, 23}, {21, 20, 68}, {7, 61, 58}, {68, 25, 24}, {20,
   19, 63}, {36, 35, 65}, {44, 59, 45}, {2, 61, 3}, {22, 68, 23}, {26,
    25, 62}, {63, 68, 20}, {58, 61, 62}, {66, 65, 69}, {39, 64,
   40}, {39, 65, 64}, {58, 62, 63}, {32, 69, 33}, {65, 38, 37}, {64,
   42, 41}, {13, 12, 60}, {64, 59, 44}, {64, 43, 42}, {63, 17,
   16}, {65, 39, 38}, {36, 65, 37}, {4, 3, 61}, {34, 33, 69}, {46, 59,
    67}, {57, 66, 69}, {66, 57, 56}, {31, 57, 69}, {65, 59, 64}, {56,
   55, 66}, {47, 46, 67}, {67, 50, 49}, {67, 49, 48}, {67, 48,
   47}, {50, 67, 51}, {46, 45, 59}, {69, 35, 34}, {53, 52, 67}, {59,
   65, 66}, {67, 52, 51}, {54, 53, 66}, {41, 40, 64}, {66, 53,
   67}, {30, 29, 28}, {28, 27, 30}, {63, 62, 68}, {69, 65, 35}, {54,
   66, 55}, {43, 64, 44}};

int main() {
    double mu=1, lambda=1;
    vector<double> k(indices.size(), 1.0);
    vector<double> nu(indices.size(), 1.0);
    vector<double> M(vertices.size(), 1.0);
    ElasticModel em(vertices, indices, k, nu, M, ElasticModel::ElasticModelType::NEOHOOKEAN);
    for(vector<unsigned>& surface: em.surfaces) {
        cout << "Chain " << endl;
        for(auto i:surface) {
            cout << i << " " << endl;
        }
    }
}
