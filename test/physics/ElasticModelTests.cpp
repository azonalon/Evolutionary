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

        em1.computeElasticForce(em1.x0, em1.g);
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


        em.computeElasticForce(em.x1, em.g);
        assertEqualsTest(0, (em.g*em.g).sum(), 0);

        psi = em.computeElasticForce(em.x0, em.g);
        f << -0.0239938, -0.0693498, 0.0394737, 0.0466718, -0.0154799, 0.022678;
        assertArrayEquals(em.g, f, 1e-4);
        assertEqualsTest(psi, 0.0623878, 1e-4);

        em.computeElasticForceDifferential(em.x0, dx, em.g);
        f << 0.148153, -0.125785, 0.0125663, 0.0110128, -0.16072, 0.114772 ;
        assertArrayEquals(em.g, f, 1e-4);

        ElasticModel em1(vertices1, indices, k, nu, M, ElasticModel::ElasticModelType::NEOHOOKEAN);
        for(unsigned i=0; i<vertices1.size(); i++) {
            em1.x0(i) = vertices2[i];
            em1.x1(i) = vertices1[i];
        }

        psi = em1.computeElasticForce(em1.x0, em1.g);
        f << -1.90296, -2.16612, 1.46365, 2.03454, 0.439309, 0.131579;
        assertArrayEquals(em1.g, f, 1e-4);
        assertEqualsTest(psi, 0.451295, 1e-4);

        em1.computeElasticForceDifferential(em1.x0, dx, em1.g);
        f << 3.0101, 1.39375, -2.58304, -1.6322, -0.427058, 0.238451;
        assertArrayEquals(em1.g, f, 1e-4);

        MatrixXd m(6,6);
        for(int i=0; i<6; i++) {
            dx = 0;
            dx(i) = 1.0;
            em1.computeElasticForceDifferential(em1.x0, dx, em1.g);
            for(int j=0; j<6;j++) {
                // assertEquals(t[i][j], em1.g.get(j, 0), 1e-4);
                // System1.out.format("%g %g %d %d\n", t[i][j], em1.g.get(j, 0), i, j);
                m(i, j) = em1.g(j);
            }
        }
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

        psi = em.computeElasticForce(em.x0, em.g);
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
            em.computeElasticForceDifferential(em.x0, dx, em.g);
            for(int j=0; j<6;j++) {
                assertEqualsTest(t[i][j], em.g(j), 1e-4);
                // System.out.format("%g %g %d %d\n", t[i][j], em.g.get(j, 0), i, j);
                // m(i, j) = em.g(j);
            }
        }
    }
    //
    //
    // void venantPiolaStress() {
    //     std::vector<double> vertices1 =  {0, 0, 0, 1, 1, 0};
    //     std::vector<double> k =  {1.0};
    //     std::vector<double> nu =  {0.33};
    //     std::vector<std::array<unsigned int,3>>indices =  {{0,1,2}};
    //     std::vector<double> M =  {1,1,1,1,1,1};
    //     ElasticModel em1 = new ElasticModel(vertices1, indices, k, nu, M);
    //     em1.venantPiolaStress(new DMatrix2x2(1, 2, 0, 3), em1.lambda[0], em1.mu[0],
    //                             em1.temp2x2A);
    //     assertEquals(new DMatrix2x2(5.88235, 18.5316, 2.25564, 26.6696),
    //                  em1.temp2x2A, 1e-4);
    // }
    //
    //  void ejmlTests() {
    //     assertEquals(normP2(new MatrixXd(new double[][] {
    //         {4},{3},{9},{2},{-5}
    //     })), 11.619, 1e-3);
    // }
    // static std::vector<double> flatten(double[][] a) {
    //     int n= a.length;
    //     int m= a[0].length;
    //     std::vector<double> f = new double[n*m];
    //     for(int i=0; i<n; i++) {
    //         for(int j=0; j<m; j++) {
    //             f[m*i +j] = a[i][j];
    //         }
    //     }
    //     return f;
    // }
    //
    // void determinant() {
    //     DMatrix2x2 m = new DMatrix2x2(1, 0, 0, 1);
    //     assertEquals(det(m), 1.0, 1e-10);
    // }
    //
    //
    static void backwardEulerTest(ElasticModel::ElasticModelType modelType,
                                  double firstVertexYOffset, Eigen::ArrayXd& finalState,
                                  double eps=0.3) {
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
        em.kDamp = 0.0;
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
    //
    //
    // void backwardEulerTestTwoTriangle() {
    //     double lambda = 0.33;
    //     double mu =  1.0;
    //     std::vector<double> vertices =  {0, 0, 0, 1, 1, 0, 1, 1};
    //     std::vector<double> k =  {mu, mu};
    //     std::vector<double> nu =  {lambda, lambda};
    //     std::vector<double> M =  {1,1,1,1,1,1,1, 1};
    //     std::vector<std::array<unsigned int,3>>indices =  {{0,1,2}, {1, 2, 3}};
    //     ElasticModel em = new ElasticModel(vertices, indices, k, nu, M);
    //
    //     // em.x0.add(1, 0, 0.7);
    //     // em.x1.add(1, 0, 0.7);
    //
    //     double dt=0.05, t=0;
    //     double u = 0.8;
    //     em.kDamp = 0.0;
    //     ImplicitODESolver.dt = dt;
    //     try {
    //         if(Files.notExists(fDir)) {
    //             Files.createDirectories(fDir);
    //         }
    //         BufferedWriter w = Files.newBufferedWriter(fDir.resolve("backwardEulerTestTwoTriangle.dat"));
    //         for(int j=0; t<10+0*j; j++) {
    //             w.write(String.format("% 8.4f", t));
    //             for(int i=0; i<vertices.length; i++) {
    //                 w.write(String.format(" % 8.4f ", em.x0.get(i)));
    //             }
    //             em.fExt.set(0, 0, u);
    //             em.fExt.set(1, 0, u);
    //             em.fExt.set(6, 0, -u);
    //             em.fExt.set(7, 0, -u);
    //             w.write(String.format("% 8.4f", em.dampPotential));
    //             w.write("\n");
    //             em.implicitEulerStep();
    //             t+=dt;
    //             ImplicitODESolver.t = t;
    //         }
    //         w.close();
    //     }
    //     catch(IOException e) {
    //         System.out.println("Error in File IO " + e);
    //     }
    // }
    //
    // static void readObj(Path file, Vector<double[]> vertices,
    //                                Vector<int[]> indices) throws IOException {
    //
    //     Scanner s = new Scanner(Files.newBufferedReader(
    //         file
    //     ));
    //     while(s.hasNextLine()) {
    //         Scanner ls = new Scanner(s.nextLine());
    //         if(!ls.hasNext()) {
    //             continue;
    //         }
    //         String token = ls.next();
    //         if("v".equals(token)) {
    //             vertices.addElement(new double[]{ls.nextDouble(), ls.nextDouble()});
    //         } else if("f".equals(token)) {
    //             indices.addElement(new int[3]);
    //             for(int i=0; i<3; i++) {
    //                 Scanner iScan = new Scanner(ls.next());
    //                 iScan.useDelimiter("/");
    //                 indices.lastElement()[i] = iScan.nextInt() - 1;
    //                 iScan.close();
    //             }
    //         }
    //         ls.close();
    //     }
    //     s.close();
    //     // for(int i=0; i<vertices.size(); i++) {
    //     //     for(int j=0; j< 2; j++) {
    //     //         System.out.print(vertices.get(i)[j] + " ");
    //     //     }
    //     //     System.out.println();
    //     // }
    //     // for(int i=0; i<indices.size(); i++) {
    //     //     for(int j=0; j< 3; j++) {
    //     //         System.out.print(indices.get(i)[j] + " ");
    //     //     }
    //     //     System.out.println();
    //     // }
    // }
    // void saveIndices(Path path, std::vector<std::array<int,3>>indices) throws IOException {
    //     BufferedWriter w = Files.newBufferedWriter(path);
    //     for(int[] face: indices) {
    //         w.write(String.format("% 4d % 4d % 4d\n", face[0], face[1], face[2]));
    //     }
    //     w.close();
    // }
    //
    // void saveIndices(Path path, Iterable<int[]> indices) throws IOException {
    //     BufferedWriter w = Files.newBufferedWriter(path);
    //     for(int[] face: indices) {
    //         w.write(String.format("% 4d % 4d % 4d\n", face[0], face[1], face[2]));
    //     }
    //     w.close();
    // }
    //
    //
    // void ball() {
    //     double lambda = 0.3;
    //     double mu =  1000.0;
    //     ballSimulation(ElasticModel.INVERTIBLE_NEOHOOKEAN, mu, lambda);
    // }
    //
    //
    // void triangleInversion() {
    //     double lambda = 0.3;
    //     double mu =  0.5;
    //     try {
    //         std::vector<double> vertices =  {0, 0, 1, 0, 0, 1};
    //         std::vector<std::array<int,3>>indices =  {{0,1,2}};
    //
    //         std::vector<double> k =  {mu} ;
    //         std::vector<double> nu =  {lambda} ;
    //         std::vector<double> M =  {1,1,1,1,1,1};
    //         ElasticModel em = new ElasticModel(vertices, indices,
    //                                            k, nu, M,
    //                                            ElasticModel.INVERTIBLE_NEOHOOKEAN,
    //                                            0.7);
    //         em.x1.set(0, 0, 1);
    //         em.x1.set(1, 0, 1);
    //         em.x0.set(em.x1);
    //         em.x2.set(em.x1);
    //         System.out.println(em.x0);
    //         System.out.println(em.x1);
    //         System.out.println(em.x2);
    //
    //         double dt=0.01, t=0;
    //         em.kDamp = 0.3;
    //         ImplicitODESolver.dt = dt;
    //         if(Files.notExists(fDir)) {
    //             Files.createDirectories(fDir);
    //         }
    //         saveIndices(fDir.resolve("triangle.indices"), indices);
    //         BufferedWriter w = Files.newBufferedWriter(
    //             fDir.resolve("triangle.dat")
    //         );
    //         for(int j=0; t<50; j++) {
    //             w.write(String.format("% 8.4f", t));
    //             for(int i=0; i<em.x0.numRows; i++) {
    //                 w.write(String.format(" % 8.4f ", em.x0.get(i)));
    //             }
    //             if(t < 5) {
    //                 em.fExt.set(0, 0, 0);
    //                 em.fExt.set(1, 0, 0);
    //                 em.fExt.set(2, 0, -0);
    //                 em.fExt.set(3, 0, 0);
    //                 em.fExt.set(4, 0, 0);
    //                 em.fExt.set(5, 0, -0);
    //             }
    //             else {
    //                 em.fExt.set(0, 0, 0);
    //                 em.fExt.set(1, 0, 0);
    //                 em.fExt.set(2, 0, 0);
    //                 em.fExt.set(3, 0, 0);
    //                 em.fExt.set(4, 0, 0);
    //                 em.fExt.set(5, 0, 0);
    //             }
    //             w.write("\n");
    //             em.implicitEulerStep();
    //             t+=dt + 0*j;
    //             ImplicitODESolver.t = t;
    //         }
    //         w.close();
    //     }
    //     catch(IOException e) {
    //         System.out.println("Error in File IO " + e);
    //     }
    //
    // }
    //
    // void ballSimulation(int model, double mu, double lambda) {
    //     try {
    //         Vector<double[]> vertices = new Vector<double[]>();
    //         Vector<int[]> indices = new Vector<int[]>();
    //         readObj(Paths.get("src/test/resources/physics/ball.obj"),
    //         vertices, indices);
    //
    //         std::vector<double> k = new double[indices.size()] ;
    //         std::vector<double> nu = new double[indices.size()] ;
    //         std::vector<double> M = new double[vertices.size()*2] ;
    //         Arrays.fill(k, mu);
    //         Arrays.fill(nu, lambda);
    //         Arrays.fill(M, 1);
    //         ElasticModel em = new ElasticModel(flatten(vertices.toArray(new double[0][])),
    //                                            indices.toArray(new int[0][]),
    //                                            k, nu, M, model, 0.3);
    //         em.x0.set(em.x1);
    //         em.x2.set(em.x1);
    //
    //         double dt=0.05, t=0;
    //         em.kDamp = 0.1;
    //         ImplicitODESolver.dt = dt;
    //         if(Files.notExists(fDir)) {
    //             Files.createDirectories(fDir);
    //         }
    //         saveIndices(fDir.resolve("ball.indices"), indices);
    //         BufferedWriter w = Files.newBufferedWriter(
    //             fDir.resolve("ball.dat")
    //         );
    //         for(int j=0; t<50; j++) {
    //             w.write(String.format("% 8.4f", t));
    //             for(int i=0; i<em.x0.numRows; i++) {
    //                 w.write(String.format(" % 8.4f ", em.x0.get(i)));
    //             }
    //             for(int i=0; i<em.x0.numRows; i++) {
    //                 if(i%2 == 1) {
    //                     em.fExt.set(i, 0, -0.2);
    //                     if(em.x0.get(i) < -5) {
    //                         em.fExt.set(i, 0, 10);
    //                     }
    //                 }
    //             }
    //             w.write("\n");
    //             // em.fExt.set(5, 0, -1);
    //             // em.fExt.set(3, 0, 1);
    //             em.implicitEulerStep();
    //             t+=dt + 0*j;
    //             ImplicitODESolver.t = t;
    //         }
    //         w.close();
    //     }
    //     catch(IOException e) {
    //         System.out.println("Error in File IO " + e);
    //     }
    // }
    //
    // void simpleForwardEuler() {
    //     double lambda = 0.33;
    //     double mu =  0.01;
    //     std::vector<double> vertices =  {0, 0, 0, 1, 1, 0};
    //     std::vector<double> k =  {mu};
    //     std::vector<double> nu =  {lambda};
    //     std::vector<std::array<int,3>>indices =  {{0,1,2}};
    //     std::vector<double> M =  {1,1,1,1,1,1};
    //     ElasticModel em = new ElasticModel(vertices, indices, k, nu, M);
    //     Random rand = new Random(0);
    //     for(int i=0; i<vertices.length; i++) {
    //         em.x0.set(i, vertices[i] + 0.0*rand.nextDouble());
    //         em.x1.set(i, vertices[i] + 0.0*rand.nextDouble());
    //     }
    //     em.x1.add(1, 0, 0.7);
    //     em.x0.add(1, 0, 0.7);
    //
    //     double dt=0.05, t=0;
    //     ImplicitODESolver.dt = dt;
    //
    //     try {
    //         if(Files.notExists(fDir)) {
    //             Files.createDirectories(fDir);
    //         }
    //         BufferedWriter w = Files.newBufferedWriter(fDir.resolve("simpleForwardEuler.dat"));
    //         for(int j=0; t<100 + 0*j; j++) {
    //             w.write(String.format("% 8.4f", t));
    //             for(int i=0; i<vertices.length; i++) {
    //                 w.write(String.format(" % 8.4f ", em.x0.get(i)));
    //             }
    //             w.write("\n");
    //             em.x2.set(em.x1);
    //             em.x1.set(em.x0);
    //             em.computeForce(em.x1, em.g);
    //             // System.out.println("Force " + em.g);
    //             add(2, em.x1, -1, em.x2, em.x0);
    //             add(dt*dt, em.g, 1, em.x0, em.x0);
    //             t+=dt;
    //         }
    //         w.close();
    //     }
    //     catch(IOException e) {
    //         System.out.println("Error in File IO " + e);
    //     }
    // }
};
int main(void) {
    ElasticModelTests::conjugateGradient();
    ElasticModelTests::compute();
    ElasticModelTests::elasticForces();
    ElasticModelTests::elasticForcesInvertibleNeoHookean();

    ArrayXd xEnd(6);
    xEnd << -0.266677, -0.256398, 0.369953, 1.65916, 0.896724, 0.297236;
    ElasticModelTests::backwardEulerTest(
        ElasticModel::ElasticModelType::NEOHOOKEAN, 0.7, xEnd, 0.5
    );
    xEnd << -0.266677, -0.256398, 0.369953, 1.65916, 0.896724, 0.297236;
    ElasticModelTests::backwardEulerTest(
        ElasticModel::ElasticModelType::INVERTIBLE_NEOHOOKEAN, 0.7, xEnd, 0.101
    );
    xEnd << 1.32266, -1.01537, 0.503395, -2.70858, -0.826054, 6.72395;
    ElasticModelTests::backwardEulerTest(
        ElasticModel::ElasticModelType::INVERTIBLE_NEOHOOKEAN, 2.0, xEnd, 0.3
    );
    return 0;
}
