link_libraries(Evolutionary)
add_executable(Test1 physics/InvertibleNeoHookeanStressDensityTests.cpp)
add_executable(Test2 physics/MathFunctionTests.cpp)
add_executable(Test3 physics/ElasticModelTests.cpp)
# add_executable(CollisionTest physics/CollisionTests.cpp)
add_executable(SurfaceTest physics/SurfaceDetectionTests.cpp)
add_executable(OptimizerTest physics/OptimizerTests.cpp)
add_executable(FluidTest physics/FluidTests.cpp)
