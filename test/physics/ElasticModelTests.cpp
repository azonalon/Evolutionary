#include "util/Assertions.hpp"
#include "physics/ElasticModel.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace Eigen;

class ElasticModelTests {
    static constexpr double mu = 0.375939849624060;
    static constexpr double lambda  = 0.729765590446705;
public:
    static void conjugateGradient() {
        std::vector<double> vertices = {0, 0, 0, 1, 1, 0};
        std::vector<double> k = {mu};
        std::vector<double> nu = {lambda};
        std::vector<double> M = {1,1,1,1,1,1};
        std::vector<std::array<unsigned int,3>> indices = {{0,1,2}};
        ElasticModel em(vertices, indices, k, nu, M,
             ElasticModel::ElasticModelType::NEOHOOKEAN);
        for(unsigned i=0; i<vertices.size(); i++) {
            em.x0(i) = vertices[i];
            em.x1(i) = vertices[i];
        }
        em.x1(1) += 0.7;
        em.x0(1) += 0.7;

        Matrix<double, 6, 6> A;
        A << 10, 0, 4, -2, 1, 0,
              0, 6, 0,  0, 5, 0,
              4, 0, 6, -1,-1, 2,
             -2, 0,-1, 10, 5, 2,
              1, 5,-1,  5,10,-1,
              0, 0, 2, 2, -1, 6;

        VectorXd b(6);
        b <<    -4, -9, -9, 10, 3, -9;
        VectorXd x0(6);
        x0 <<   0, 0, 0, 0, 0, 0;
        VectorXd xCorrect(6);
        xCorrect <<   0.0223104, -2.05611, -0.784498, 0.876219, 0.667328, -1.41935;
        ArrayXd x(6);
        auto computeLhs = [&](auto& m, auto& d) -> void {
            d = A * m.matrix();
        };
        em.conjugateGradientSolve(b, x0, computeLhs, x);
        assertArrayEquals(x, xCorrect, 1e-3);

        Matrix<double, 6, 6> B;
        B  << 10,-6,2,6,-6,-1,-6,2,3,3,-1,-6,2,3,8,-2,-3,-6,6,3,-2,-8,2,-5,-6,-1,-3,2,-6,-1,-1,-6,-6,-5,-1,-4;
        b  << -10,-3,0,4,-1,-5;
        x0 << 0, 0, 0, 0, 0, 0;
        xCorrect  << 0.0115413,-0.631618,0.193112,-0.663991,-1.42397,-2.29857;
        // x = new MatrixXd(6, 1);
        auto computeLhs2 = [&](auto& m, auto& d) -> void {
            d = B*m.matrix();
        };
        em.conjugateGradientSolve(b, x0, computeLhs2, x);
        assertArrayNotEquals(x, xCorrect, 1e-3);
    }


    static void compute() {
        std::vector<double> vertices1 =  {0, 0, 0, 1, 1, 0};
        std::vector<double> vertices2 =  {0, 0, 0, 1, 1, 1};
        std::vector<double> k =   {mu};
        std::vector<double> nu =   {lambda};
        std::vector<std::array<unsigned int,3>>indices =  {{0,1,2}};
        std::vector<double> M =   {1,1,1,1,1,1};
        ElasticModel em1 (vertices1, indices, k, nu, M,
             ElasticModel::ElasticModelType::NEOHOOKEAN);
        ElasticModel em2 (vertices2, indices, k, nu, M,
             ElasticModel::ElasticModelType::NEOHOOKEAN);
        Matrix2d Bm1; Bm1 << -1, -1,  0, 1;
        Matrix2d Bm2; Bm2 <<  0, -1, -1, 1;
        assertArrayEquals(Bm1, em1.Bm[0], 1e-12);
        assertEqualsTest(em1.lambda[0], 0.729766, 1e-5);
        assertEqualsTest(em1.mu[0], 0.37594, 1e-5);
        assertArrayEquals(em2.Bm[0], Bm2, 1e-12);
        for(unsigned i=0; i<vertices1.size(); i++) {
            em1.x0(i) = vertices1[i];
            em1.x1(i) = vertices1[i];
        }
        Matrix2d id; id << 1, 0, 0, 1;
        em1.venantPiolaStress(id, 1, 1, em1.temp2x2P);
        assertEqualsTest(0, em1.temp2x2P.norm(), 0);

        em1.g = 0;
        em1.computeGradient(em1.x0, em1.g);
        em1.g = -em1.g;
        assertEqualsTest(0, (em1.g*em1.g).sum(), 0);
    }
    //
    //
    static void elasticForces() {
        std::vector<double> vertices1 =   {0, 0, 0, 1, 1, 0};
        // std::vector<double> vertices2 =   {0, -1, 2, 2, 1, -1};
        std::vector<double> vertices2 =   {0, 0.7, 0, 1, 1, 0};
        std::vector<double> k =   {mu};
        std::vector<double> nu =   {lambda};
        std::vector<std::array<unsigned int,3>>indices =  {{0,1,2}};
        std::vector<double> M =   {1,1,1, 1,1,1};
        ElasticModel em(vertices1, indices, k, nu, M,
            ElasticModel::ElasticModelType::VENANTKIRCHHOFF);
        for(unsigned i=0; i<vertices1.size(); i++) {
            em.x0(i)= vertices2[i];
            em.x1(i)= vertices1[i];
        }
        ArrayXd dx(vertices1.size()), f(vertices1.size());
        dx << -.05, .1, -.2, .3, .4, .5;
        double psi;


        em.g = 0;
        em.computeElasticGradient(em.x1, em.g);
        em.g = -em.g;
        assertEqualsTest(0, (em.g*em.g).sum(), 0);

        em.g = 0;
        psi = em.computeElasticGradient(em.x0, em.g);
        em.g = -em.g;
        f << -0.0239938, -0.0693498, 0.0394737, 0.0466718, -0.0154799, 0.022678;
        assertArrayEquals(em.g, f, 1e-4);
        assertEqualsTest(psi, 0.0623878, 1e-4);

        em.g = 0;
        em.computeElasticDifferential(em.x0, dx, em.g);
        em.g = -em.g;
        f << 0.148153, -0.125785, 0.0125663, 0.0110128, -0.16072, 0.114772 ;
        assertArrayEquals(em.g, f, 1e-4);

        ElasticModel em1(vertices1, indices, k, nu, M, ElasticModel::ElasticModelType::NEOHOOKEAN);
        for(unsigned i=0; i<vertices1.size(); i++) {
            em1.x0(i) = vertices2[i];
            em1.x1(i) = vertices1[i];
        }

        em1.g = 0;
        psi = em1.computeElasticGradient(em1.x0, em1.g);
        em1.g = -em1.g;
        f << -1.90296, -2.16612, 1.46365, 2.03454, 0.439309, 0.131579;
        assertArrayEquals(em1.g, f, 1e-4);
        assertEqualsTest(psi, 0.451295, 1e-4);

        em1.g = 0;
        em1.computeElasticDifferential(em1.x0, dx, em1.g);
        em1.g = -em1.g;
        f << 3.0101, 1.39375, -2.58304, -1.6322, -0.427058, 0.238451;
        assertArrayEquals(em1.g, f, 1e-4);

        // MatrixXd m(6,6);
        // for(int i=0; i<6; i++) {
        //     dx = 0;
        //     dx(i) = 1.0;
        //     em.g = 0;
        //     em1.computeElasticForceDifferential(em1.x0, dx, em1.g);
        //     for(int j=0; j<6;j++) {
        //         // assertEquals(t[i][j], em1.g.get(j, 0), 1e-4);
        //         // System1.out.format("%g %g %d %d\n", t[i][j], em1.g.get(j, 0), i, j);
        //         m(i, j) = em1.g(j);
        //     }
        // }
    }

    static void elasticForcesInvertibleNeoHookean() {
        std::vector<double>vertices1 =  {0, 0, 0, 1, 1, 1};
        // std::vector<double>vertices2 =  {0, -1, 2, 2, 1, -1};
        std::vector<double>vertices2 =  {1.2, 1.1, 0.1, 1.05, 1.06, 1.07};
        std::vector<double>k =  {mu};
        std::vector<double>nu =  {lambda};
        std::vector<std::array<unsigned int,3>>indices =  {{0,1,2}};
        std::vector<double> M =  {1,1,1,1,1,1};
        ArrayXd dx(vertices1.size()), f(vertices1.size());
        double psi;
        dx << -.05, .1, -.2, .3, .4, .5;


        ElasticModel em(vertices1, indices, k, nu, M,
                ElasticModel::ElasticModelType::INVERTIBLE_NEOHOOKEAN , 0.3);
        for(unsigned i=0; i<vertices1.size(); i++) {
            em.x0(i) =  vertices2[i];
            em.x1(i) =  vertices1[i];
        }

        em.g = 0;
        psi = em.computeElasticGradient(em.x0, em.g);
        em.g = -em.g;
        f <<    -0.046988, -4.7444, 0.420897, -0.681655, -0.373909,  5.42606;
        assertEqualsTest(1.68113, psi, 1e-4);
        assertArrayEquals(em.g, f, 1e-4);
        double t[6][6] = {{-0.356551, 0.646852, 0.506009, -4.82607, -0.149458,
                          4.17922}, {0.646852, -6.77856, 4.19242, -0.872466, -4.83927,
                          7.65102}, {0.506009, 4.19242, -0.911141, -0.133457,
                          0.405132, -4.05896}, {-4.82607, -0.872466, -0.133457, -0.417634,
                          4.95953, 1.2901}, {-0.149458, -4.83927, 0.405132,
                          4.95953, -0.255674, -0.120252}, {4.17922, 7.65102, -4.05896,
                          1.2901, -0.120252, -8.94112}};

        Matrix2d F,dF,dP,t1;
        F  <<  0.96,-1.1,0.02,-0.05;
        dF <<  0.05,-0.07,-0.07,0.02;
        dP <<  0,0,0,0;
        t1 << -0.163915, -0.675827, -1.51878, -1.19337;
        InvertibleNeoHookeanModel neo(0.3);
        neo.computeStressDifferential(F, dF, 0.729766, 0.37594, dP);
        assertArrayEquals(t1, dP, 0.0001);


        // MatrixXd m(6,6);
        for(int i=0; i<6; i++) {
            dx = 0;
            dx(i) = 1.0;
            em.g = 0;
            em.computeElasticDifferential(em.x0, dx, em.g);
            em.g = -em.g;
            for(int j=0; j<6;j++) {
                assertEqualsTest(t[i][j], em.g(j), 1e-4);
                // System.out.format("%g %g %d %d\n", t[i][j], em.g.get(j, 0), i, j);
                // m(i, j) = em.g(j);
            }
        }
    }
    //
    //

    static void backwardEulerTest(ElasticModel::ElasticModelType modelType,
                                  double firstVertexYOffset, Eigen::ArrayXd& finalState,
                                  double eps=0.3, double kDamp=0) {
        std::vector<double> vertices =  {0, 0, 0, 1, 1, 0};
        std::vector<double> k =  {mu};
        std::vector<double> nu =  {lambda};
        std::vector<double> M =  {1,1,1,1,1,1};
        std::vector<std::array<unsigned int,3>>indices =  {{0,1,2}};
        ElasticModel em(vertices, indices, k, nu, M,
            modelType, eps);

        em.x0(1)+= firstVertexYOffset;
        em.x1(1)+= firstVertexYOffset;
        em.x2(1)+= firstVertexYOffset;

        double t=0, tEnd = 10;
        em.kDamp = kDamp;
        em.dt = 0.001;
        std::ofstream w;
        w.open("backwardEulerTest.dat");
        w.precision(4);
        w.width(16);
        for(int j=0; t<tEnd; j++) {
            w << t;
            for(unsigned i=0; i<vertices.size(); i++) {
                w << " " << em.x2(i) << " ";
            }
            em.fExt(0) = 0*cos(M_PI*t);
            w << std::endl;
            em.implicitEulerStep();
            t+= em.dt + 0*j;
        }
        w.close();
        assertArrayEquals(em.x2, finalState, 1e-3);
    }

};
int main(void) {
    ElasticModelTests::conjugateGradient();
    ElasticModelTests::compute();
    ElasticModelTests::elasticForces();
    ElasticModelTests::elasticForcesInvertibleNeoHookean();

    // ArrayXd xEnd(6);

    // xEnd <<  0.559664, 0.409414, 0.113522, -1.01994, 0.326814, 3.61053;
    // ElasticModelTests::backwardEulerTest(
    //     ElasticModel::ElasticModelType::INVERTIBLE_NEOHOOKEAN, 2.0, xEnd, 0.3, 0.1
    // );

    // xEnd << -0.266677, -0.256398, 0.369953, 1.65916, 0.896724, 0.297236;
    // ElasticModelTests::backwardEulerTest(
    //     ElasticModel::ElasticModelType::NEOHOOKEAN, 0.7, xEnd, 0.5
    // );
    // xEnd << -0.266677, -0.256398, 0.369953, 1.65916, 0.896724, 0.297236;
    // ElasticModelTests::backwardEulerTest(
    //     ElasticModel::ElasticModelType::INVERTIBLE_NEOHOOKEAN, 0.7, xEnd, 0.101
    // );
    // xEnd << 1.32266, -1.01537, 0.503395, -2.70858, -0.826054, 6.72395;
    // ElasticModelTests::backwardEulerTest(
    //     ElasticModel::ElasticModelType::INVERTIBLE_NEOHOOKEAN, 2.0, xEnd, 0.3
    // );
    return 0;
}
