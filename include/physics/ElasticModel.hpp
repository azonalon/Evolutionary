// import static physics.InvertibleNeoHookeanModel;
#pragma once
#include <Eigen/Dense>
#include "ImplicitODESolver.hpp"
#include "InvertibleNeoHookeanModel.hpp"
#include <Eigen/StdVector>
/**
 * Finite Element elasticity model.
 */
class ElasticModel: public ImplicitODESolver {
public:
    // matrices for triangles
    // reference triangle matrices
    unsigned n, m; // number of points and triangles
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> Bm;
    Eigen::Matrix2d temp2x2A, temp2x2B, temp2x2P, temp2x2D, H, P, F, dF, dP;
    const Eigen::Matrix2d id  =  Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d nId = -Eigen::Matrix2d::Identity();
    enum ElasticModelType {
        NEOHOOKEAN,
        VENANTKIRCHHOFF,
        INVERTIBLE_NEOHOOKEAN
    };

    double invertibleEpsilon = 0.3;

    ElasticModelType model = NEOHOOKEAN;
    InvertibleNeoHookeanModel neo;
    // std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> Te; // nx3 indices for each triangle into point vector
    Eigen::Matrix<unsigned int, Eigen::Dynamic, 3, Eigen::RowMajor> Te;
    std::vector<double> W; // reference triangle volumes
    std::vector<double> mu; // first lame coefficient for each triangle
    std::vector<double> lambda; // second lame coefficient

    unsigned triangleCount() {
        return m;
    }
    unsigned vertexCount() {
        return n/2;
    }
    static double computeCollisionPenaltyForce(
        unsigned int iPoint,unsigned int  i, unsigned int j,unsigned int  k,
        const Eigen::ArrayXd& x, Eigen::ArrayXd& dest);

    static void computeCollisionPenaltyForceDifferential(
        unsigned int iPoint,unsigned int  i, unsigned int j,unsigned int  k,
        const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest);

    // /**
    //  * Generates a finite element elastic model solver.
    //  * Vertices are stored in a single vector (x1, y1, x2, y2..., xn, yn)
    //  * triangle indices are ((t11, t12, t13), (t21, t22, t23), ..., (tm1, tm2, tm3))
    //  * the tij specify offsets to the x coordinates of the vertex stored in vertices
    //  */
    // ElasticModel(std::vector<double> vertices,  const std::vector<std::array<int,3>>& triangles,
    //                     std::vector<double> k, std::vector<double> nu, std::vector<double> M) {
    //     ElasticModel(vertices, triangles, k, nu, M, NEOHOOKEAN);
    // }

    /**
     * Generates a finite element elastic model solver.
     * Vertices are stored in a single vector (x1, y1, x2, y2..., xn, yn)
     * triangle indices are ((t11, t12, t13), (t21, t22, t23), ..., (tm1, tm2, tm3))
     * the tij specify offsets to the x coordinates of the vertex stored in vertices
     * TODO: implement class for different elastic models and their respective options
     */
    template<class ...Ts>
    ElasticModel(const std::vector<double>& vertices, const std::vector<std::array<unsigned int,3>>& triangles,
                const std::vector<double>& k, const std::vector<double>& nu, const std::vector<double>& M,
                ElasticModelType model, double eps=0.5):
                        ImplicitODESolver(vertices.size(), M), n(vertices.size()), m(triangles.size()),
                        Bm(m), model(model), neo(eps), Te(m, 3), W(m), mu(m), lambda(m)
                        {

        assert(triangles.size() == k.size());
        assert(nu.size() == k.size());
        assert(M.size() == vertices.size());
        assert(vertices.size() % 2 == 0);
        for(unsigned i=0; i<m; i++) {
            Te.row(i) <<
                triangles[i][0],
                triangles[i][1],
                triangles[i][2]
            ;
        }
        precompute(vertices, k, nu);
        for(unsigned i=0; i<vertices.size(); i++) {
            x0(i) = vertices[i];
            x1(i) = vertices[i];
            x2(i) = vertices[i];
        }
    }


    void precompute(const std::vector<double>& vertices,
                    const std::vector<double>& K, const std::vector<double>& nu) {
        for(unsigned l=0; l<m; l++){
            unsigned int i=2*Te(l, 0), j=2*Te(l, 1), k=2*Te(l, 2);
            temp2x2A <<
                  vertices[i + 0] - vertices[k + 0], vertices[j + 0] - vertices[k + 0],
                  vertices[i + 1] - vertices[k + 1], vertices[j + 1] - vertices[k + 1];

            W[l] = std::abs(temp2x2A.determinant()/2.0);
            assert(Math::invert(temp2x2A, Bm[l]));
            lambda[l] = K[l]*nu[l]/(1+nu[l])/(1-2*nu[l]);
            mu[l] = K[l]/2/(1+nu[l]);
        }
    }


    double computeElasticForce(const Eigen::ArrayXd& x, Eigen::ArrayXd& dest) {
        dest = 0;
        double stressEnergy=0;
        for(unsigned l=0; l<m; l++){
            unsigned int i=2*Te(l, 0), j=2*Te(l, 1), k=2*Te(l, 2);
            temp2x2A <<
                  x(i + 0) - x(k + 0), x(j + 0) - x(k + 0),
                  x(i + 1) - x(k + 1), x(j + 1) - x(k + 1);
            temp2x2B = temp2x2A*Bm[l];
            if(model == NEOHOOKEAN)
                stressEnergy += W[l]*neoHookeanStress(temp2x2B, lambda[l], mu[l], temp2x2A);
            else if(model == VENANTKIRCHHOFF)
                stressEnergy += W[l]*venantPiolaStress(temp2x2B, lambda[l], mu[l], temp2x2A);
            else
                stressEnergy += W[l]*neo.computeStressTensor(temp2x2B, lambda[l], mu[l], temp2x2A);
            temp2x2B = temp2x2A * Bm[l].transpose();
            addForceMatrixToVector(temp2x2B, dest, i, j, k, l);

            // TODO: collision testing, basically you should provide a set points,
            // this triangle is possibly colliding with. for now we check every
            // point (brute force)
            // TODO: this does not work if one point is contained in multiple triangles
            for(unsigned n=0; n<m; n++){
                if(n==l) {
                    continue;
                }
                for(unsigned jj=0; jj<3;jj++) {
                    unsigned int iPoint = 2*Te(n,jj);
                    // stressEnergy += computeCollisionPenaltyForce(iPoint, i, j, k, x, dest);
                }
            }
        }
        return stressEnergy;
    }


    /**
     * Calculates the change of force df for small displacements around x.
     * f ~ f(x) + K dx = f(x) + df(x, dx)
     * df = K dx; where K ist the Stiffness Matrix.
     * The stiffness Matrix itself is not evaluated. But rather the response
     * of the Piola Kirchhoff Stress due to a small displacement.
     * This is useful for calculating the Newton direction for optimization
     * deltax = K^-1 f
     */
    void computeElasticForceDifferential(const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx,
                                         Eigen::ArrayXd& dest) {
        dest = 0;
        for(unsigned l=0; l<m; l++){
            unsigned int i=2*Te(l, 0), j=2*Te(l, 1), k=2*Te(l, 2);
            temp2x2A <<
                  x(i + 0) - x(k + 0), x(j + 0) - x(k + 0),
                  x(i + 1) - x(k + 1), x(j + 1) - x(k + 1);
            temp2x2B <<
                  dx(i + 0) - dx(k + 0), dx(j + 0) - dx(k + 0),
                  dx(i + 1) - dx(k + 1), dx(j + 1) - dx(k + 1);
            F = temp2x2A * Bm[l];
            dF = temp2x2B * Bm[l];
            if(model == NEOHOOKEAN)
                neoHookeanStressDifferential(F, dF, lambda[l], mu[l], dP);
            else if(model == VENANTKIRCHHOFF)
                venantPiolaStressDifferential(F, dF, lambda[l], mu[l], dP);
            else
                neo.computeStressDifferential(F, dF, lambda[l], mu[l], dP);
            temp2x2B = dP*Bm[l].transpose();
            addForceMatrixToVector(temp2x2B, dest, i, j, k, l);
            for(unsigned n=0; n<m; n++){
                if(n==l) {
                    continue;
                }
                for(unsigned jj=0; jj<3;jj++) {
                    unsigned int iPoint = 2*Te(n,jj);
                    // computeCollisionPenaltyForceDifferential(iPoint, i, j, k, x, dx, dest);
                }
            }
        }
    }

    virtual double computeForce(const Eigen::ArrayXd& x, Eigen::ArrayXd& dest) override {
        return computeElasticForce(x, dest);
    }
    virtual void computeForceDifferential(const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx,
                                               Eigen::ArrayXd& dest) override {
        computeElasticForceDifferential(x, dx, dest);
    }

    inline static double normP2Squared(Eigen::Matrix2d& m) {
        return
            m(0,0)*m(0,0) +
            m(0,1)*m(0,1) +
            m(1,0)*m(1,0) +
            m(1,1)*m(1,1);
    }

    inline void addForceMatrixToVector(Eigen::Matrix2d& H, Eigen::ArrayXd& f,
                                             int i, int j, int k, int l) {
        f(i + 0) += -W[l]*H(0,0);
        f(i + 1) += -W[l]*H(1,0);
        f(j + 0) += -W[l]*H(0,1);
        f(j + 1) += -W[l]*H(1,1);
        f(k + 0) += +W[l]*H(0,0) +W[l]*H(0,1);
        f(k + 1) += +W[l]*H(1,0) +W[l]*H(1,1);
    }

    double venantPiolaStress(Eigen::Matrix2d& F,
                                 double lambda, double mu,
                                 Eigen::Matrix2d& dest) {
        double psi=0;
        temp2x2P << nId;
        temp2x2P += F.transpose()*F;
        psi += mu*normP2Squared(temp2x2P)/4.0;
        double tr = temp2x2P.trace();
        double k = lambda * tr/2.0;
        psi += tr*tr*lambda/8.0;
        temp2x2P = mu*temp2x2P + k*id;
        dest = F*temp2x2P;
        return psi;
    }


    double neoHookeanStress(Eigen::Matrix2d& F,
                                 double lambda, double mu,
                                 Eigen::Matrix2d& dest) {
        temp2x2P = F.transpose()*F;
        double I1 = temp2x2P.trace();
        // double J = det(F);
        double J = F.determinant();
        assert(J > 0 && "Negative jacobian not supported for neo hookean!");
        double logJ = std::log(J);
        Math::invertTranspose(F, temp2x2P);
        dest = mu*F + (lambda*logJ - mu)*temp2x2P;
        return mu*(I1/2.0 - 1 - logJ) + lambda/2.0*logJ*logJ;
    }

    void neoHookeanStressDifferential(Eigen::Matrix2d& F,Eigen::Matrix2d& dF,
                                 double lambda, double mu,
                                 Eigen::Matrix2d& dest) {
        // double J = det(F);
        double J = F.determinant();
        assert(J > 0 && "Negative jacobian not supported for neo hookean!");
        double logJ = std::log(J);

        Math::invertTranspose(F, temp2x2P);
        temp2x2D = temp2x2P * dF.transpose();
        double trIFdF = temp2x2D.trace();
        dest = temp2x2D*temp2x2P;
        dest = mu*dF + (mu - lambda*logJ)*dest;
        dest = dest + lambda*trIFdF*temp2x2P;
    }

    void venantPiolaStressDifferential(Eigen::Matrix2d& F, Eigen::Matrix2d& dF,
                                 double lambda, double mu,
                                 Eigen::Matrix2d& destDP) {
        temp2x2P << nId;
        temp2x2P += F.transpose()*F;
        double k = lambda * temp2x2P.trace() / 2.0;
        temp2x2P = mu*temp2x2P + k*id;
        destDP = dF*temp2x2P;

        temp2x2P = dF.transpose()*F;
        temp2x2P += F.transpose()*dF;
        double dk = lambda * temp2x2P.trace() / 2.0;
        temp2x2P = mu*temp2x2P + dk*id;
        destDP += F*temp2x2P;
    }
};
