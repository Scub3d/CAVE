#pragma once

#include <vector>
#include "../common/structs.h"

namespace Cave
{
    class ObjMesh
    {
    private:
        std::vector<Vertex> _vertices;
        std::vector<uint32_t> _indices;

    public:
        ObjMesh(const char *objFilepath, const char *mtlFilepath);
        ~ObjMesh();

        std::vector<Vertex> GetMeshVertices() const { return _vertices; }
        std::vector<uint32_t> GetMeshIndices() const { return _indices; }
    };

} // namespace Cave
