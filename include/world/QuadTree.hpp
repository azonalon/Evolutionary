#include "physics/ElasticModel.hpp"
class QuadTreeNode {
    QuadTreeNode* qA=std::nullptr;
    QuadTreeNode* qB=std::nullptr;
    QuadTreeNode* qC=std::nullptr;
    QuadTreeNode* qD=std::nullptr;

    int nPoints=0;
    int vertexIndex;
    int surface;
    int surfaceIndex;
}

class QuadTree {
    QuadTree(ElasticModel m, std::array<double, 4> bounds);
    std::vector<QuadTreeNode> nodes;
    void InsertNode(int vertexIndex, int surface, int surfaceIndex);
}