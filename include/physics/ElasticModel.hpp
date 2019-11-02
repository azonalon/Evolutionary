// import static physics.InvertibleNeoHookeanModel;
#pragma once
#include <Eigen/Dense>
#include "ImplicitODESolver.hpp"
#include "InvertibleNeoHookeanModel.hpp"
#include "StableNeoHookean.hpp"
#include "CollisionObjects.hpp"
#include <Eigen/StdVector>
#include <map>
/**
 * Finite Element elasticity model.
 */
extern double strengthSelfCollision;
extern double muSelfFriction;

class ElasticModel: public ImplicitODESolver {
public:
    // matrices for triangles
    // reference triangle matrices
    unsigned n; // twice the number of vertices 
    unsigned m; // number of triangles
    std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> Bm;
    Eigen::Matrix2d temp2x2A, temp2x2B, temp2x2P, temp2x2D, H, P, F, dF, dP;
    const Eigen::Matrix2d id  =  Eigen::Matrix2d::Identity();
    const Eigen::Matrix2d nId = -Eigen::Matrix2d::Identity();
    // surfaces contains indices to the outer edges for each disconnected 
    // object
    std::vector<std::vector<unsigned int>> surfaces;
    // selfcollisionlist contains in the first index the vertex which collided
    // with the face defined by the two vertices corresponding to the last
    // two indices
    std::vector<std::array<unsigned, 3>> selfCollisionList;
    // contains the closest surface edge for each edge (distance between edges
    // is not really defined, but it is an approximation).
    std::map<std::array<unsigned, 2>, std::array<unsigned,2>> closestSurfaceFromEdge;
    std::vector<unsigned> surfaceFromVertex;
    std::map<std::array<unsigned, 2>, int> isSurfaceEdge;
    std::function<void(const std::vector<std::array<unsigned, 3>>&)> collisionListPopulatedHook =
        [](auto collisionList) { return; };

    enum ElasticModelType {
        NEOHOOKEAN,
        VENANTKIRCHHOFF,
        INVERTIBLE_NEOHOOKEAN,
        STABLE_NEOHOOKEAN
    };

    double invertibleEpsilon = 0.3;
    double kFluid = 1;
    std::vector<CollisionObject*> collisionObjects;


    ElasticModelType model = NEOHOOKEAN;
    // std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> Te; // nx3 indices for each triangle into point vector
    InvertibleNeoHookeanModel neo;
    Eigen::Matrix<unsigned int, Eigen::Dynamic, 3, Eigen::RowMajor> Te;
    std::vector<double> W; // reference triangle volumes
    std::vector<double> mu; // first lame coefficient for each triangle
    std::vector<double> lambda; // second lame coefficient
    std::vector<double> S; // stretch factors

    unsigned triangleCount() {
        return m;
    }
    unsigned vertexCount() {
        return n/2;
    }

    void setSurfaceForce(unsigned iObject, unsigned iSurface, double force);

    double computeCollisionPenaltyGradient(const Eigen::ArrayXd& x, Eigen::ArrayXd& dest);

    void computeCollisionPenaltyGradientDifferential(const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest);

    double computeFluidFrictionGradient(const Eigen::ArrayXd& x, Eigen::ArrayXd& dest);

    void computeFluidFrictionGradientDifferential(const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx, Eigen::ArrayXd& dest);

    std::array<unsigned, 3> closestSurfaceFromPoint(unsigned iPoint, unsigned iSurface, const Eigen::ArrayXd& x); 

      // /**
      //  * Generates a finite element elastic model solver.
      //  * Vertices are stored in a single vector (x1, y1, x2, y2..., xn, yn)
      //  * triangle indices are ((t11, t12, t13), (t21, t22, t23), ..., (tm1,
      //  tm2, tm3))
      //  * the tij specify offsets to the x coordinates of the vertex stored in
      //  vertices
      //  */
      // ElasticModel(std::vector<double> vertices,  const
      // std::vector<std::array<int,3>>& triangles,
      //                     std::vector<double> k, std::vector<double> nu,
      //                     std::vector<double> M) {
      //     ElasticModel(vertices, triangles, k, nu, M, NEOHOOKEAN);
      // }

      /**
       * Generates a finite element elastic model solver.
       * Vertices are stored in a single vector (x1, y1, x2, y2..., xn, yn)
       * triangle indices are ((t11, t12, t13), (t21, t22, t23), ..., (tm1, tm2,
       * tm3)) the tij specify offsets to the x coordinates of the vertex stored
       * in vertices
       * TODO: implement class for different elastic models and their respective
       * options
       */
      template <class... Ts>
      ElasticModel(const std::vector<double> &vertices,
                   const std::vector<std::array<unsigned int, 3>> &triangles,
                   const std::vector<double> &_mu,
                   const std::vector<double> &_lambda,
                   const std::vector<double> &M, ElasticModelType model,
                   double eps = 0.5)
          : ImplicitODESolver(vertices.size(), M), n(vertices.size()),
            m(triangles.size()), Bm(m), model(model), neo(eps), Te(m, 3), W(m),
            mu(m), lambda(m), S(m), surfaceFromVertex(n / 2) {

        assert(triangles.size() == _mu.size());
        assert(_mu.size() == _lambda.size());
        assert(M.size() == vertices.size());
        assert(vertices.size() % 2 == 0);
        for(unsigned i=0; i<m; i++) {
            Te.row(i) <<
                triangles[i][0],
                triangles[i][1],
                triangles[i][2]
            ;
            this->mu[i] = _mu[i];
            this->lambda[i] = _lambda[i];
            this->S[i] = 1;
            // lambda[l] = K[l]*nu[l]/(1+nu[l])/(1-2*nu[l]);
            // mu[l] = K[l]/2/(1+nu[l]);
        }
        precompute(vertices);
        collisionPrecompute(vertices);
        for(unsigned i=0; i<vertices.size(); i++) {
            x0(i) = vertices[i];
            x1(i) = vertices[i];
            x2(i) = vertices[i];
        }
    }

    void collisionPrecompute(const std::vector<double>& vertices);

    void precompute(const std::vector<double>& vertices) {
        for(unsigned l=0; l<m; l++){
            unsigned int i=2*Te(l, 0), j=2*Te(l, 1), k=2*Te(l, 2);
            temp2x2A <<
                  vertices[i + 0] - vertices[k + 0], vertices[j + 0] - vertices[k + 0],
                  vertices[i + 1] - vertices[k + 1], vertices[j + 1] - vertices[k + 1];

            W[l] = std::abs(temp2x2A.determinant()/2.0);
            assert(Math::invert(temp2x2A, Bm[l]));
        }
    }


    double computeElasticGradient(const Eigen::ArrayXd& x, Eigen::ArrayXd& dest) {
        double stressEnergy=0;
        for(unsigned l=0; l<m; l++){
            unsigned int i=2*Te(l, 0), j=2*Te(l, 1), k=2*Te(l, 2);
            temp2x2A <<
                  x(i + 0) - x(k + 0), x(j + 0) - x(k + 0),
                  x(i + 1) - x(k + 1), x(j + 1) - x(k + 1);
            temp2x2B = temp2x2A*S[l]*Bm[l];
            if(model == NEOHOOKEAN)
                stressEnergy += W[l]*neoHookeanStress(temp2x2B, lambda[l], mu[l], temp2x2A);
            else if(model == VENANTKIRCHHOFF)
                stressEnergy += W[l]*venantPiolaStress(temp2x2B, lambda[l], mu[l], temp2x2A);
            else if(model == STABLE_NEOHOOKEAN)
                stressEnergy += W[l]*StableNeoHookeanModel::computeStressTensor(temp2x2B, lambda[l], mu[l], temp2x2A);
            else
                stressEnergy += W[l]*neo.computeStressTensor(temp2x2B, lambda[l], mu[l], temp2x2A);
            temp2x2B = temp2x2A * S[l]*Bm[l].transpose();
            addGradientMatrixToVector(temp2x2B, dest, i, j, k, l);
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
     *
     * !!Make sure to zero out dest because the calculations only add to dest
     */
    void computeElasticDifferential(const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx,
                                         Eigen::ArrayXd& dest) {
        for(unsigned l=0; l<m; l++){
            unsigned int i=2*Te(l, 0), j=2*Te(l, 1), k=2*Te(l, 2);
            temp2x2A <<
                  x(i + 0) - x(k + 0), x(j + 0) - x(k + 0),
                  x(i + 1) - x(k + 1), x(j + 1) - x(k + 1);
            temp2x2B <<
                  dx(i + 0) - dx(k + 0), dx(j + 0) - dx(k + 0),
                  dx(i + 1) - dx(k + 1), dx(j + 1) - dx(k + 1);
            F = temp2x2A * S[l]*Bm[l];
            dF = temp2x2B * S[l]*Bm[l];
            if(model == NEOHOOKEAN)
                neoHookeanStressDifferential(F, dF, lambda[l], mu[l], dP);
            else if(model == VENANTKIRCHHOFF)
                venantPiolaStressDifferential(F, dF, lambda[l], mu[l], dP);
            else if(model == STABLE_NEOHOOKEAN)
                StableNeoHookeanModel::computeStressDifferential(F, dF, lambda[l], mu[l], dP);
            else
                neo.computeStressDifferential(F, dF, lambda[l], mu[l], dP);
            temp2x2B = dP*S[l]*Bm[l].transpose();
            addGradientMatrixToVector(temp2x2B, dest, i, j, k, l);
        }
    }

    void populateSelfCollisionList(const Eigen::ArrayXd& x);

    virtual double computeGradient(const Eigen::ArrayXd& x, Eigen::ArrayXd& dest) override {
      double E = computeElasticGradient(x, dest);
      E += computeCollisionPenaltyGradient(x, dest);
    //   E += computeFluidFrictionGradient(x, dest);
      return E;
    }
    virtual void computeDifferential(const Eigen::ArrayXd& x, const Eigen::ArrayXd& dx,
                                               Eigen::ArrayXd& dest) override {
        computeElasticDifferential(x, dx, dest);
        computeCollisionPenaltyGradientDifferential(x, dx, dest);
        // computeFluidFrictionGradientDifferential(x, dx, dest);
        return;
    }


    virtual void precomputeStep(const Eigen::ArrayXd& x) override {
        populateSelfCollisionList(x);
        // TODO actually precompute stress tensors etc.
    };


    inline void addGradientMatrixToVector(Eigen::Matrix2d& H, Eigen::ArrayXd& f,
                                             unsigned i, unsigned j, unsigned k, unsigned l) {
        f(i + 0) += +W[l]*H(0,0);
        f(i + 1) += +W[l]*H(1,0);
        f(j + 0) += +W[l]*H(0,1);
        f(j + 1) += +W[l]*H(1,1);
        f(k + 0) += -W[l]*H(0,0) -W[l]*H(0,1);
        f(k + 1) += -W[l]*H(1,0) -W[l]*H(1,1);
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
    inline static double venantPiolaStress(Eigen::Matrix2d& F,
                                 double lambda, double mu,
                                 Eigen::Matrix2d& dest) {
        Eigen::Matrix2d E = 0.5 *(F.transpose() * F - Eigen::Matrix2d::Identity());
        dest = F*(2*mu*E + lambda* E.trace() * Eigen::Matrix2d::Identity());
        return mu*E.squaredNorm() + 0.5*lambda*E.trace()*E.trace();
    }

    inline static void venantPiolaStressDifferential(Eigen::Matrix2d& F, Eigen::Matrix2d& dF,
                                 double lambda, double mu,
                                 Eigen::Matrix2d& destDP) {
        Eigen::Matrix2d E = 0.5 *(F.transpose() * F - Eigen::Matrix2d::Identity());
        Eigen::Matrix2d dE = 0.5 *(dF.transpose() * F + F.transpose()*dF);
        destDP = dF*(2*mu*E + lambda*E.trace()*Eigen::Matrix2d::Identity())
                    + F*(2*mu*dE + lambda*dE.trace()*Eigen::Matrix2d::Identity() ) ;
    }
    };
