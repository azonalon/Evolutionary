#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <cassert>

inline void assertArrayEquals(Eigen::MatrixXd a, Eigen::MatrixXd b, double delta) {
    if(!a.isApprox(b, delta)) {
        std::cout
        << "Expected Array:\n"
        << a << std::endl
        << "Got Array\n"
        << b << std::endl;
        assert(false && "AssertArrayEquals fail.");
    }
}

inline void assertArrayNotEquals(Eigen::MatrixXd a, Eigen::MatrixXd b, double delta) {
    if(a.isApprox(b, delta)) {
        std::cout
        << "---------------------------------------"
        << a << std::endl
        << "---------------------------------------"
        << b << std::endl;
        assert(false && "AssertArrayNotEquals fail.");
    }
}

inline void assertTrue(std::string message, bool value) {
    if(!value) {
        std::stringstream s;
        s << "ERROR: " << message << std::endl << "FILE: " << __FILE__ << ":" << __LINE__ << std::endl;
        std::runtime_error(s.str());
    }
}
inline void assertEquals(double a, double b, double eps) {
    if(std::abs(a-b) > eps) {
        assert(false && "AssertEquals fail.");
    }
}
inline void assertNotEquals(double a, double b, double eps) {
    if(std::abs(a-b) < eps) {
        assert(false && "AssertNotEquals fail.");
    }
}
