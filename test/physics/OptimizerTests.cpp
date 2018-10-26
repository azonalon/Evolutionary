#include "physics/ElasticModel.hpp"
// #define STBIWDEF inline
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define RGBA 4
#include "stbi_image_write.h"
#include <fstream>

using Eigen::Dynamic;
using Eigen::RowMajor;
// void map(Eigen::MatrixXd m, Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor>& data)  {
//     double max = m.maxCoeff();
//     double min = m.minCoeff();
//     auto ma = (m.array() - min)/(max - min);
//
//     for(int i=0; i<m.rows(); i++)
//         for(int j=0; j<m.cols(); j++) {
//         data(i,4*j + 0) = 254*ma(i,j);
//         data(i,4*j + 1) = 254*ma(i,j);
//         data(i,4*j + 2) = 254*ma(i,j);
//         data(i,4*j + 3) = 254;
//     }
// }


int main(void){
    double mu =1;
    double lambda = 1;
    std::vector<double> vertices =  {0, 0, 0, 1, 1, 0};
    std::vector<double> k =  {mu};
    std::vector<double> nu =  {lambda};
    std::vector<double> M =  {1,1,1,1,1,1};
    std::vector<std::array<unsigned int,3>>indices =  {{0,1,2}};
    ElasticModel em(vertices, indices, k, nu, M,
        ElasticModel::ElasticModelType::INVERTIBLE_NEOHOOKEAN, 0.3);
    em.g = 0;
    em.kDamp = 0.0;
    em.dt = std::numeric_limits<double>::max();
    double energy;

    static const unsigned N = 264;
    double smin = -2, srange = 4;
    Eigen::MatrixXd m(N, N);
    // Eigen::Matrix<unsigned char, Dynamic, Dynamic, RowMajor> data(N, 4*N);
    for(unsigned i=0; i < N; i++) {
        for(unsigned j=0; j < N; j++) {
            for(int k=0; k<em.x0.size(); k+=2) {
                em.x0[k+0] = vertices[k+0]*(smin + (double)i/(double)N * srange);
                em.x0[k+1] = vertices[k+1]*(smin + (double)j/(double)N * srange);
            }
            em.x1 = em.x0;
            m(i,j) = em.computeOptimizeGradient(em.x0, em.g);
        }
    }
    std::ofstream f;
    f.open("test.tiff");
    f << m;
    f.close();
    // map(m, data);
    // stbi_write_png("test.png", N, N, RGBA, &data(0,0), 4*N);
}
