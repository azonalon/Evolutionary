#include <Eigen/Dense>
#include <iostream>
#include <cassert>
#pragma once

#define assertEqualsTest(a, e, epsilon) _assertEqualsTest(__FILE__, __func__, __LINE__, a, e, epsilon)
#define assertArrayEquals(a, e, epsilon) _assertArrayEquals(__FILE__, __func__, __LINE__, a, e, epsilon)
// #define assertTrueTest(a, e, epsilon)   _assertTrueTest(__FILE__, __func__, __LINE__, a, e, epsilon)

inline bool _assertArrayEquals(const char *file, const char *function, int line,
                                Eigen::MatrixXd b, Eigen::MatrixXd a, double delta) {
    if(a.isApprox(b, delta)) {
		printf("%s, line %d: check passed\n", file, line);
		return true;
    } else {
		printf("%s, line %d: Actual Array is not within %g of expected value.\n", file, line, delta);
        std::cout << a << std::endl
        << "Got Array\n"
        << b << std::endl;
        throw std::runtime_error("Assertion error.");
		return false;
    }
}

inline void assertArrayNotEquals(Eigen::MatrixXd b, Eigen::MatrixXd a, double delta) {
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

inline bool _assertEqualsTest(const char *file, const char *function, int line, double a, double e, double epsilon) {
	if (fabs(a - e) <= epsilon) {
		printf("%s, line %d: check passed\n", file, line);
		return true;
	} else {
		printf("%s, line %d: Actual value %g is not within %g of expected value %g.\n", file, line, a, epsilon, e);
        throw std::runtime_error("Assertion error.");
		return false;
	}
}

// inline bool _assertTrueTest(const char *file, const char *function, int line, bool v) {
// 	if (v == true) {
// 		printf("%s, line %d: check passed\n", file, line);
// 		return true;
// 	} else {
// 		printf("%s, line %d: Result should not have been false.\n", file, line, v, true);
//         throw std::runtime_error("Assertion error.");
// 		return false;
// 	}
// }
