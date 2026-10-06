// The model's surface as triangles, for the march (THEORY 10.10): each
// pixel's distance along its ray to the nearest triangle and the shading
// normal there, from a raster pass with the march's own camera. The march
// then draws the model's true shape at any zoom, painted from the flow as
// before; the solver's voxels stay one choice away (Surface::Voxel).
// Internal to the renderer.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "windoa/context.hpp"
#include "windoa/mesh.hpp"

namespace windoa::render {

// The march's camera basis, as its push constants carry it.
struct RasterCamera {
    float eye[4];   // xyz, tan(fov / 2)
    float fwd[4];   // xyz, aspect
    float right[4]; // xyz, -
    float up[4];    // xyz, -
};

// Corner data for the pass: per corner a position and a normal (vec4 each).
// The normal averages the faces round a corner's position whose normals lie
// within kCreaseDeg of the corner's own face (weighted by area), so curved
// surfaces shade smoothly and edges sharper than that stay sharp.
inline constexpr float kCreaseDeg = 40.0f;
std::vector<float> mesh_corners(const geometry::Mesh& m);

class MeshRaster {
  public:
    explicit MeshRaster(Context& ctx);
    ~MeshRaster();
    MeshRaster(const MeshRaster&) = delete;
    MeshRaster& operator=(const MeshRaster&) = delete;

    // The triangles in lattice cells. Waits for the graphics queue (frames
    // in flight read the old ones); uploaded by the next record().
    void set_mesh(const geometry::Mesh& m);
    bool empty() const { return corners_ == 0; }
    // The output size (the renderer's); waits for the graphics queue when
    // it changes.
    void resize(std::uint32_t width, std::uint32_t height);
    // Record the pass into a graphics command buffer, then copy its result
    // into `dst` (at least width x height vec4): per pixel, row 0 at the
    // top, the normal and the distance (1e30 where no triangle), ready for
    // a compute shader to read.
    void record(VkCommandBuffer cmd, const RasterCamera& cam, const Buffer& dst);

  private:
    void create_targets();

    Context& ctx_;
    VkDevice device_;
    // in creation order: destroyed images -> buffers -> pool -> pipeline ...
    ShaderModuleHandle vs_, fs_;
    SetLayoutHandle set_layout_;
    PipelineLayoutHandle layout_;
    PipelineHandle pipeline_;
    DescriptorPoolHandle pool_;
    VkDescriptorSet set_ = VK_NULL_HANDLE; // freed with the pool
    std::unique_ptr<Buffer> corner_buf_, staging_;
    std::uint32_t corners_ = 0;
    bool upload_pending_ = false;
    std::uint32_t width_ = 0, height_ = 0;
    MemoryHandle colour_memory_;
    ImageHandle colour_;
    ImageViewHandle colour_view_;
    MemoryHandle depth_memory_;
    ImageHandle depth_;
    ImageViewHandle depth_view_;
};

} // namespace windoa::render
