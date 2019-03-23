#pragma once
#include <Eigen/Dense>
#include "physics/CollisionObjects.hpp"
#include "physics/ElasticModel.hpp"

class Phonons {
public:
  ElasticModel *model;
  Eigen::Map<Eigen::MatrixXd> m;
  Phonons(ElasticModel *model, unsigned i0, unsigned n1, unsigned n2)
      : model(model),
        m(Eigen::Map<Eigen::MatrixXd, 0>(&model->S[i0], n1, n2)){};
  void update(double t);
};